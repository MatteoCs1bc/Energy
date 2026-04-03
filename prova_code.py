import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import njit

# ==========================================
# CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(page_title="Simulatore Mix Energetico PRO", layout="wide")

# ==========================================
# PESI GEOGRAFICI CURVE MEDIE
# ==========================================
PV_WEIGHTS_NORD = {
    'Lombardia orientale, area Brescia_NORD': 0.2956,
    'Veneto centrale, area Padova_NORD': 0.2313,
    'Emilia-Romagna orientale, area Ferrara,pianura_NORD': 0.2213,
    'Piemonte meridionale, area Cuneo_NORD': 0.1874,
    'Friuli-Venezia Giulia, area Udine_NORD': 0.0644,
}

PV_WEIGHTS_SUD = {
    'Puglia, area Lecce_SUD': 0.3241,
    'Sicilia interna, area Caltanissetta,Enna_SUD': 0.2117,
    'Lazio meridionale, area Latina_SUD': 0.1982,
    'Sardegna, area Oristano,Campidano_SUD': 0.1330,
    'Campania interna, area Benevento_SUD': 0.1330,
}

WIND_WEIGHTS_NORD = {
    'Crinale savonese entroterra ligure_NORD': 0.6020,
    'Appennino emiliano, area Monte Cimone_NORD': 0.2239,
    'Piemonte sud-occidentale , Cuneese_NORD': 0.0945,
    'Veneto orientale , Delta del Po_NORD': 0.0647,
    'Valle d’Aosta , area alpina_NORD': 0.0149,
}

WIND_WEIGHTS_SUD = {
    'Puglia, area Foggia,Daunia_SUD': 0.3093,
    'Sicilia occidentale, area Trapani_SUD': 0.2267,
    'Campania, area Benevento,Avellino_SUD': 0.1950,
    'Basilicata, area Melfi,Potenza_SUD': 0.1489,
    'Calabria, area Crotone,Catanzaro_SUD': 0.1201,
}

DEFAULT_PV_NORD_SHARE = 0.4800
DEFAULT_WIND_NORD_SHARE = 0.0163

# ==========================================
# FUNZIONI DI SUPPORTO
# ==========================================
def _serie_pesata(df, pesi_colonne, scala=1.0, clip_upper=1.0):
    colonne_mancanti = [col for col in pesi_colonne if col not in df.columns]
    if colonne_mancanti:
        raise KeyError("Nel dataset mancano le colonne richieste: " + ", ".join(colonne_mancanti))

    serie = sum(pd.to_numeric(df[col], errors='coerce').fillna(0.0) * peso for col, peso in pesi_colonne.items())
    serie = (serie / scala).clip(lower=0.0)
    if clip_upper is not None:
        serie = serie.clip(upper=clip_upper)
    return serie.astype(float)


def _mappa_profilo_annuale_su_indice(profilo_orario, indice_target):
    profilo = profilo_orario.copy()
    profilo.index = pd.to_datetime(profilo.index)

    chiavi_sorgente = list(zip(profilo.index.month, profilo.index.day, profilo.index.hour))
    mappa = {chiave: valore for chiave, valore in zip(chiavi_sorgente, profilo.values)}

    valori = []
    for ts in indice_target:
        chiave = (ts.month, ts.day, ts.hour)
        if chiave in mappa:
            valori.append(mappa[chiave])
        elif ts.month == 2 and ts.day == 29:
            valori.append(mappa.get((2, 28, ts.hour), mappa.get((3, 1, ts.hour), 0.0)))
        else:
            valori.append(0.0)

    return pd.Series(valori, index=indice_target, dtype=float)


@st.cache_data
def leggi_gme(file_gme):
    df_gme = pd.read_excel(file_gme, engine='openpyxl')

    # 1. Ricerca dinamica della colonna dei consumi
    col_volumi = None
    for col in df_gme.columns:
        if str(col).lower() in ['fabbisogno', 'totale', 'volume', 'load', 'consumi', 'mwh']:
            col_volumi = col
            break
    
    # Fallback standard: prende la 3° colonna come nel GME originale
    if col_volumi is None:
        if len(df_gme.columns) >= 3:
            col_volumi = df_gme.columns[2]
        else:
            col_volumi = df_gme.columns[-1]

    # 2. Pulizia della formattazione europea (es. 1.000,50 -> 1000.50)
    if df_gme[col_volumi].dtype == 'object':
        df_gme[col_volumi] = df_gme[col_volumi].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)

    # Trasforma tutto in numeri e cancella le celle testuali vuote/sporche
    valori_numerici = pd.to_numeric(df_gme[col_volumi], errors='coerce').dropna()

    if valori_numerici.empty or valori_numerici.sum() <= 0:
        raise ValueError(f"Impossibile leggere i consumi dalla colonna '{col_volumi}'. Verifica il file GME.")

    # 3. BYPASS ASSOLUTO DELLE DATE: Generiamo un calendario sintetico perfetto sulle N ore
    ore_totali = len(valori_numerici)
    if ore_totali == 8784:
        idx = pd.date_range(start="2024-01-01 00:00", periods=ore_totali, freq='h') # Anno bisestile
    else:
        idx = pd.date_range(start="2023-01-01 00:00", periods=ore_totali, freq='h') # Anno standard
        
    df_pulito = pd.DataFrame({'Fabbisogno_MW': valori_numerici.values}, index=idx)
    return df_pulito


@st.cache_data
def carica_profili_rinnovabili(file_fotovoltaico, file_eolico):
    df_pv = pd.read_csv(file_fotovoltaico)
    df_pv['time'] = pd.to_datetime(df_pv['time'], errors='coerce')
    df_pv = df_pv.dropna(subset=['time']).copy()
    df_pv.set_index('time', inplace=True)

    df_wind = pd.read_csv(file_eolico)
    df_wind['time'] = pd.to_datetime(df_wind['time'], errors='coerce')
    df_wind = df_wind.dropna(subset=['time']).copy()
    df_wind.set_index('time', inplace=True)

    profili = {
        'pv_nord': pd.Series(_serie_pesata(df_pv, PV_WEIGHTS_NORD, scala=1000.0, clip_upper=1.0).values, index=df_pv.index, name='pv_nord'),
        'pv_sud': pd.Series(_serie_pesata(df_pv, PV_WEIGHTS_SUD, scala=1000.0, clip_upper=1.0).values, index=df_pv.index, name='pv_sud'),
        'wind_nord': pd.Series(_serie_pesata(df_wind, WIND_WEIGHTS_NORD, scala=1.0, clip_upper=1.0).values, index=df_wind.index, name='wind_nord'),
        'wind_sud': pd.Series(_serie_pesata(df_wind
