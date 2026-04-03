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
def leggi_gme_radar(file_gme):
    try:
        df_gme = pd.read_excel(file_gme, engine='openpyxl', header=None)
        miglior_serie = None
        max_somma = -1
        
        for col in df_gme.columns:
            s = df_gme[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            s_num = pd.to_numeric(s, errors='coerce').dropna()
            
            if len(s_num) >= 8760:
                somma_corrente = s_num.sum()
                if somma_corrente > max_somma:
                    max_somma = somma_corrente
                    ore_anno = 8784 if len(s_num) >= 8784 else 8760
                    miglior_serie = s_num.iloc[:ore_anno]
        
        if miglior_serie is not None and max_somma > 0:
            idx = pd.date_range(start="2023-01-01 00:00", periods=len(miglior_serie), freq='h')
            return pd.DataFrame({'Fabbisogno_MW': miglior_serie.values}, index=idx), False
            
    except Exception:
        pass
    
    ore = 8760
    idx = pd.date_range(start="2023-01-01 00:00", periods=ore, freq='h')
    t = np.arange(ore)
    giorno_notte = 10000 * np.sin(2 * np.pi * t / 24 - np.pi/2)
    stagione = 5000 * np.sin(2 * np.pi * t / 8760 - np.pi/2)
    fabbisogno_finto = np.clip(25000 + giorno_notte + stagione, a_min=15000, a_max=60000)
    return pd.DataFrame({'Fabbisogno_MW': fabbisogno_finto}, index=idx), True


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
        'wind_sud': pd.Series(_serie_pesata(df_wind, WIND_WEIGHTS_SUD, scala=1.0, clip_upper=1.0).values, index=df_wind.index, name='wind_sud'),
    }
    return profili


@st.cache_data
def carica_dati_v2(file_fotovoltaico, file_gme, file_eolico, quota_pv_nord, quota_eolico_nord):
    df_gme, usato_fallback = leggi_gme_radar(file_gme)
    profili = carica_profili_rinnovabili(file_fotovoltaico, file_eolico)

    quota_pv_nord = float(quota_pv_nord)
    quota_eolico_nord = float(quota_eolico_nord)

    profilo_pv = (profili['pv_nord'] * quota_pv_nord) + (profili['pv_sud'] * (1.0 - quota_pv_nord))
    profilo_wind = (profili['wind_nord'] * quota_eolico_nord) + (profili['wind_sud'] * (1.0 - quota_eolico_nord))

    df_completo = df_gme.copy()
    df_completo['Fattore_Capacita_PV'] = _mappa_profilo_annuale_su_indice(profilo_pv, df_completo.index)
    df_completo['Fattore_Capacita_Wind'] = _mappa_profilo_annuale_su_indice(profilo_wind, df_completo.index)

    return df_completo.ffill(), usato_fallback


# ==========================================
# 2. SIMULAZIONE FISICA (Numba) - Rete
# ==========================================
@njit
def simula_rete_light_fast(produzione_pv, produzione_wind, fabbisogno,
                           pv_mw, wind_mw, nucleare_mw, bess_mwh, bess_mw, gas_mw,
                           hydro_fluente_mw, hydro_bacino_mw, hydro_bacino_max_mwh, hydro_inflow_mw,
                           efficienza_bess=0.9):
    ore = len(fabbisogno)
    soc_corrente = bess_mwh * 0.5
    soc_hydro = hydro_bacino_max_mwh * 0.5

    prod_pv_array = produzione_pv * pv_mw
    prod_wind_array = produzione_wind * wind_mw
    potenza_nucleare_costante = nucleare_mw * 1

    gas_usato_totale = 0.0
    deficit_totale = 0.0
    overgen_totale = 0.0
    hydro_dispatched_totale = 0.0
    bess_scarica_totale = 0.0

    sqrt_eff = np.sqrt(efficienza_bess)

    for t in range(ore):
        soc_hydro += hydro_inflow_mw
        if soc_hydro > hydro_bacino_max_mwh:
            soc_hydro = hydro_bacino_max_mwh

        generazione_base = prod_pv_array[t] + prod_wind_array[t] + hydro_fluente_mw + potenza_nucleare_costante
        bilancio_netto = generazione_base - fabbisogno[t]

        if bilancio_netto > 0:
            spazio_libero_batteria = bess_mwh - soc_corrente
            potenza_assorbibile_max = spazio_libero_batteria / sqrt_eff
            potenza_carica_effettiva = min(bilancio_netto, bess_mw, potenza_assorbibile_max)
            soc_corrente += potenza_carica_effettiva * sqrt_eff
            overgen_totale += (bilancio_netto - potenza_carica_effettiva)
        else:
            energia_richiesta = abs(bilancio_netto)

            potenza_scarica_bess = min(energia_richiesta, bess_mw)
            energia_out_bess = potenza_scarica_bess / sqrt_eff
            if soc_corrente >= energia_out_bess:
                soc_corrente -= energia_out_bess
                energia_richiesta -= potenza_scarica_bess
                bess_scarica_totale += potenza_scarica_bess
            else:
                energia_disp_bess = soc_corrente * sqrt_eff
                soc_corrente = 0.0
                energia_richiesta -= energia_disp_bess
                bess_scarica_totale += energia_disp_bess

            if energia_richiesta > 0:
                potenza_scarica_hydro = min(energia_richiesta, hydro_bacino_mw)
                if soc_hydro >= potenza_scarica_hydro:
                    soc_hydro -= potenza_scarica_hydro
                    energia_richiesta -= potenza_scarica_hydro
                    hydro_dispatched_totale += potenza_scarica_hydro
                else:
                    hydro_dispatched_totale += soc_hydro
                    energia_richiesta -= soc_hydro
                    soc_hydro = 0.0

            if energia_richiesta > 0:
                uso_gas = min(energia_richiesta, gas_mw)
                gas_usato_totale += uso_gas
                deficit_totale += (energia_richiesta - uso_gas)

    return gas_usato_totale, deficit_totale, overgen_totale, hydro_dispatched_totale, bess_scarica_totale


# ==========================================
# NUOVO MOTORE H2: CO-OTTIMIZZAZIONE ORARIA (Numba)
# ==========================================
@njit
def estrai_curva_overgen_oraria(produzione_pv, produzione_wind, fabbisogno,
                                pv_mw, wind_mw, nucleare_mw, bess_mwh, bess_mw,
                                hydro_fluente_mw, hydro_bacino_max_mwh, hydro_inflow_mw, efficienza_bess=0.9):
    ore = len(fabbisogno)
    soc_corrente = bess_mwh * 0.5
    soc_hydro = hydro_bacino_max_mwh * 0.5
    prod_pv_array = produzione_pv * pv_mw
    prod_wind_array = produzione_wind * wind_mw
    potenza_nucleare_costante = nucleare_mw * 1
    sqrt_eff = np.sqrt(efficienza_bess)
    
    overgen_array = np.zeros(ore)
    
    for t in range(ore):
        soc_hydro += hydro_inflow_mw
        if soc_hydro > hydro_bacino_max_mwh:
            soc_hydro = hydro_bacino_max_mwh
            
        generazione_base = prod_pv_array[t] + prod_wind_array[t] + hydro_fluente_mw + potenza_nucleare_costante
        bilancio_netto = generazione_base - fabbisogno[t]
        
        if bilancio_netto > 0:
            spazio_libero_batteria = bess_mwh - soc_corrente
            potenza_assorbibile_max = spazio_libero_batteria / sqrt_eff
            potenza_carica_effettiva = min(bilancio_netto, bess_mw, potenza_assorbibile_max)
            soc_corrente += potenza_carica_effettiva * sqrt_eff
            overgen_array[t] = bilancio_netto - potenza_carica_effettiva
        else:
            energia_richiesta = abs(bilancio_netto)
            potenza_scarica_bess = min(energia_richiesta, bess_mw)
            energia_out_bess = potenza_scarica_bess / sqrt_eff
            if soc_corrente >= energia_out_bess:
                soc_corrente -= energia_out_bess
                energia_richiesta -= potenza_scarica_bess
            else:
                energia_disp_bess = soc_corrente * sqrt_eff
                soc_corrente = 0.0
                energia_richiesta -= energia_disp_bess

            if energia_richiesta > 0:
                potenza_scarica_hydro = min(energia_richiesta, 12000.0) 
                if soc_hydro >= potenza_scarica_hydro:
                    soc_hydro -= potenza_scarica_hydro
                else:
                    soc_hydro = 0.0
                    
    return overgen_array


@njit
def co_ottimizza_h2_rinnovabile(overgen_array, vre_profile_1GW, target_mwh_el,
                                capex_elc_mw, capex_batt_mwh, costo_annuo_vre_gw, crf):
    """
    Trova la combinazione ottimale di (Elettrolizzatore, Rinnovabile Dedicata, Batterie Dedicate)
    SIMULTANEAMENTE al recupero del curtailment di rete.
    """
    miglior_costo = 1e15
    best_elc_gw = 0.0
    best_vre_gw = 0.0
    best_batt_gwh = 0.0
    best_recupero_mwh = 0.0
    
    # Costruiamo una griglia di ricerca dinamica (per evitare loop infiniti)
    # Se servono 10 TWh, base_elc = ~1.1 GW. Cerchiamo fino a 6x.
    elc_base = target_mwh_el / 8760.0
    ore_eq_vre = np.sum(vre_profile_1GW)
    vre_base = target_mwh_el / ore_eq_vre if ore_eq_vre > 0 else 10.0
    
    elc_steps = np.linspace(elc_base * 0.5, elc_base * 6.0, 20)
    vre_steps = np.linspace(0.0, vre_base * 2.0, 20)
    batt_steps = np.linspace(0.0, vre_base * 4.0, 15)
    
    ore_tot = len(overgen_array)
    
    for vre_gw in vre_steps:
        costo_vre = vre_gw * costo_annuo_vre_gw
        for elc_gw in elc_steps:
            costo_elc = elc_gw * 1000.0 * capex_elc_mw * crf
            for batt_gwh in batt_steps:
                costo_batt = batt_gwh * 1000.0 * capex_batt_mwh * crf
                
                costo_tot = costo_vre + costo_elc + costo_batt
                # Pruning: se l'impianto costa già più del "miglior costo" trovato finora, non serve simularlo!
                if costo_tot >= miglior_costo:
                    continue
                    
                # Simula le 8760 ore con questa specifica configurazione
                soc = 0.0
                batt_mw = batt_gwh * 1000.0
                elc_mw = elc_gw * 1000.0
                eff = 0.95
                energia_assorbita = 0.0
                recupero_grid = 0.0
                
                for t in range(ore_tot):
                    p_vre = vre_profile_1GW[t] * vre_gw
                    p_grid = overgen_array[t]
                    
                    # Disponibilità immediata
                    p_avail = p_vre + p_grid
                    
                    # Scarica della batteria se l'energia diretta non basta a saturare l'elettrolizzatore
                    p_discharge = 0.0
                    if p_avail < elc_mw and soc > 0:
                        p_discharge = min(soc * eff, batt_mw, elc_mw - p_avail)
                        
                    p_tot = p_avail + p_discharge
                    
                    # Cut-off Elettrolizzatore (15%)
                    if p_tot >= elc_mw * 0.15:
                        p_elc = min(p_tot, elc_mw)
                        energia_assorbita += p_elc
                        
                        # Calcolo attribuzione: quanta energia proviene dalla rete?
                        if p_vre >= p_elc:
                            usato_grid = 0.0
                            p_excess = p_vre - p_elc + p_grid
                        else:
                            if p_vre + p_grid >= p_elc:
                                usato_grid = p_elc - p_vre
                                p_excess = p_grid - usato_grid
                            else:
                                usato_grid = p_grid
                                p_excess = 0.0
                                
                        recupero_grid += usato_grid
                        soc -= (p_discharge / eff)
                        
                        # Carica la batteria con l'eccesso totale
                        if p_excess > 0 and soc < batt_gwh * 1000.0:
                            charge = min(p_excess, batt_mw, (batt_gwh * 1000.0 - soc)/eff)
                            soc += charge * eff
                    else:
                        # Elettrolizzatore spento. Tutta l'energia va in batteria.
                        if p_avail > 0 and soc < batt_gwh * 1000.0:
                            charge = min(p_avail, batt_mw, (batt_gwh * 1000.0 - soc)/eff)
                            soc += charge * eff
                            
                # Fine dell'anno. Ha raggiunto il target?
                if energia_assorbita >= target_mwh_el:
                    if costo_tot < miglior_costo:
                        miglior_costo = costo_tot
                        best_elc_gw = elc_gw
                        best_vre_gw = vre_gw
                        best_batt_gwh = batt_gwh
                        best_recupero_mwh = recupero_grid
                        
    # Fallback d'emergenza se la griglia non trova nulla
    if best_elc_gw == 0.0:
        best_elc_gw = elc_base * 3.5
        best_vre_gw = vre_base * 1.5
        best_batt_gwh = 0.0
        best_recupero_mwh = 0.0
        
    return float(best_elc_gw), float(best_vre_gw), float(best_batt_gwh), float(best_recupero_mwh)


@njit
def ottimizza_h2_nucleare(overgen_array, target_mwh_el, capex_elc_mw, cfd_nuc, cf_nuc, crf):
    miglior_costo = 1e15
    best_elc_gw = 0.0
    best_recupero_mwh = 0.0
    
    elc_base = target_mwh_el / 8760.0
    ore_tot = len(overgen_array)
    
    for elc_gw in np.linspace(elc_base * 0.5, elc_base * 5.0, 50):
        costo_elc = elc_gw * 1000.0 * capex_elc_mw * crf
        elc_mw = elc_gw * 1000.0
        
        energia_assorbita = 0.0
        recupero_grid = 0.0
        p_nuc_piatto = elc_mw * cf_nuc
        
        for t in range(ore_tot):
            p_grid = overgen_array[t]
            p_avail = p_nuc_piatto + p_grid
            
            p_elc = min(p_avail, elc_mw)
            energia_assorbita += p_elc
            
            if p_nuc_piatto >= p_elc:
                usato_grid = 0.0
            else:
                usato_grid = p_elc - p_nuc_piatto
                
            recupero_grid += usato_grid
            
        if energia_assorbita >= target_mwh_el:
            costo_tot = costo_elc + ((energia_assorbita - recupero_grid) * cfd_nuc)
            if costo_tot < miglior_costo:
                miglior_costo = costo_tot
                best_elc_gw = elc_gw
                best_recupero_mwh = recupero_grid
                
    if best_elc_gw == 0.0:
        best_elc_gw = elc_base / cf_nuc
        best_recupero_mwh = 0.0
        
    return float(best_elc_gw), float(best_recupero_mwh)


# ==========================================
# 3. MOTORE DI CALCOLO SEPARATO (RETE MAIN)
# ==========================================
@st.cache_data
def simula_tutti_scenari_fisici(array_pv, array_wind, array_fabbisogno):
    scenari_pv_gw = [40, 50, 80, 100, 150]
    scenari_wind_gw = [10, 20, 30, 60, 90]
    scenari_bess_gwh = [10, 30, 50, 150, 300, 400]
    scenari_nuc_gw = [0, 2, 5, 10, 15, 20, 25, 30]

    GAS_CAPACITA_FISSA_MW = 50000
    BESS_POTENZA_FISSA_MW = 50000
    HYDRO_FLUENTE_MW = 2500.0
    HYDRO_BACINO_MW = 12000.0
    HYDRO_BACINO_MAX_MWH = 5000000.0
    HYDRO_INFLOW_MW = 2850.0

    risultati_fisici = []

    for pv in scenari_pv_gw:
        for wind in scenari_wind_gw:
            for bess in scenari_bess_gwh:
                for nuc in scenari_nuc_gw:
                    gas_mwh, def_mwh, over_mwh, hydro_disp_mwh, bess_out_mwh = simula_rete_light_fast(
                        array_pv, array_wind, array_fabbisogno,
                        pv * 1000.0, wind * 1000.0, nuc * 1000.0, bess * 1000.0, BESS_POTENZA_FISSA_MW, GAS_CAPACITA_FISSA_MW,
                        HYDRO_FLUENTE_MW, HYDRO_BACINO_MW, HYDRO_BACINO_MAX_MWH, HYDRO_INFLOW_MW
                    )

                    risultati_fisici.append({
                        'PV_GW': pv, 'Wind_GW': wind, 'BESS_GWh': bess, 'Nuc_GW': nuc,
                        'gas_mwh': gas_mwh, 'deficit_mwh': def_mwh, 'overgen_mwh': over_mwh,
                        'hydro_disp_mwh': hydro_disp_mwh, 'bess_scarica_mwh': bess_out_mwh
                    })

    return risultati_fisici


def applica_economia_e_trova_ottimo(risultati_fisici, df_completo, mercato):
    fabbisogno_tot_mwh = df_completo['Fabbisogno_MW'].sum()
    
    if fabbisogno_tot_mwh <= 0:
        raise ValueError("Il fabbisogno totale calcolato è 0. C'è un problema nella lettura del GME.")

    ore_eq_pv = df_completo['Fattore_Capacita_PV'].sum()
    ore_eq_wind = df_completo['Fattore_Capacita_Wind'].sum()
    hydro_fluente_tot_mwh = 2500.0 * len(df_completo)

    LCA_EMISSIONI = {'pv': 45.0, 'wind': 11.0, 'hydro': 24.0, 'nuc': 12.0, 'bess': 50.0, 'gas': 550.0}

    wacc = mercato.get('wacc_bess', 0.05)
    vita = mercato.get('bess_vita', 15)
    opex_f_rate = mercato.get('bess_opex_fix', 0.015)

    if wacc > 0:
        crf = (wacc * (1 + wacc) ** vita) / ((1 + wacc) ** vita - 1)
    else:
        crf = 1 / vita

    storia = []

    for r in risultati_fisici:
        pv_mw = r['PV_GW'] * 1000.0
        wind_mw = r['Wind_GW'] * 1000.0
        nuc_mw = r['Nuc_GW'] * 1000.0
        bess_mwh = r['BESS_GWh'] * 1000.0

        costo_pv = (pv_mw * ore_eq_pv) * mercato['cfd_pv']
        costo_wind = (wind_mw * ore_eq_wind) * mercato['cfd_wind']
        costo_hydro = (hydro_fluente_tot_mwh + r['hydro_disp_mwh']) * mercato['gas_eur_mwh']
        costo_nuc = (nuc_mw * 1 * 8760) * mercato['cfd_nuc']

        capex_investimento = bess_mwh * mercato['bess_capex']
        costo_bess = (capex_investimento * crf) + (capex_investimento * opex_f_rate)

        lcos = costo_bess / r['bess_scarica_mwh'] if r['bess_scarica_mwh'] > 0 else 0.0

        energia_vre_totale = (pv_mw * ore_eq_pv) + (wind_mw * ore_eq_wind)
        quota_vre = energia_vre_totale / fabbisogno_tot_mwh

        costo_base_integr = mercato['costo_base_integrazione'] * (quota_vre ** 2)
        potenza_media_carico = fabbisogno_tot_mwh / 8760
        rapporto_bess = (r['BESS_GWh'] * 1000) / potenza_media_carico
        sconto_bess = min(0.5, rapporto_bess / 5.0)

        costo_sistema_totale = energia_vre_totale * (costo_base_integr * (1 - sconto_bess))

        costo_gas = r['gas_mwh'] * mercato['gas_eur_mwh']
        costo_blackout = r['deficit_mwh'] * mercato['voll']

        costo_bolletta = (costo_pv + costo_wind + costo_hydro + costo_nuc + costo_bess + costo_gas + costo_blackout + costo_sistema_totale) / fabbisogno_tot_mwh
        percentuale_gas = (r['gas_mwh'] / fabbisogno_tot_mwh) * 100

        emi_tot = ((pv_mw * ore_eq_pv) * LCA_EMISSIONI['pv'] +
                   (wind_mw * ore_eq_wind) * LCA_EMISSIONI['wind'] +
                   (hydro_fluente_tot_mwh + r['hydro_disp_mwh']) * LCA_EMISSIONI['hydro'] +
                   (nuc_mw * 1 * 8760) * LCA_EMISSIONI['nuc'] +
                   r['bess_scarica_mwh'] * LCA_EMISSIONI['bess'] +
                   r['gas_mwh'] * LCA_EMISSIONI['gas'])
        carbon_intensity = emi_tot / fabbisogno_tot_mwh

        storia.append({
            'Configurazione': f"{r['PV_GW']}PV|{r['Wind_GW']}W|{r['BESS_GWh']}B|{r['Nuc_GW']}N",
            'PV_GW': r['PV_GW'], 'Wind_GW': r['Wind_GW'], 'BESS_GWh': r['BESS_GWh'], 'Nuc_GW': r['Nuc_GW'],
            'Costo_Bolletta': costo_bolletta,
            'Carbon_Intensity': carbon_intensity,
            'Gas_%': percentuale_gas,
            'LCOS_BESS': lcos,
            'Overgen_TWh': r['overgen_mwh'] / 1e6,
            'gas_mwh': r['gas_mwh']
        })

    df_risultati = pd.DataFrame(storia)

    # Scudo Budget 5%
    min_costo = df_risultati['Costo_Bolletta'].min()
    soglia_prezzo = min_costo * 1.05

    scenari_ok = df_risultati[df_risultati['Costo_Bolletta'] <= soglia_prezzo]
    
    if scenari_ok.empty:
        miglior_config = df_risultati.sort_values(by='Carbon_Intensity').iloc[0].to_dict()
    else:
        miglior_config = scenari_ok.sort_values(by='Carbon_Intensity').iloc[0].to_dict()

    return miglior_config, df_risultati


# ==========================================
# 4. INTERFACCIA UTENTE (STREAMLIT)
# ==========================================
try:
    st.title("⚡ Ottimizzatore Mix Energetico e Decarbonizzazione (BETA)")
    st.markdown("Scopri l'equilibrio tra Rinnovabili, Batterie e Nucleare valutando le emissioni dell'intero ciclo di vita.")

    @st.dialog("📖 Come funziona questo simulatore?")
    def mostra_spiegazione():
        st.markdown("""
        **Benvenuto nel Simulatore di Mix Energetico 1.0 di CS1BC!**
        per smanettare coi parametri clicca le freccette in alto a sinistra e aggiorni il risultato della funzione obiettivo
        Cos'è CS1BC? è un collettivo strafigo! unbelclima.it

        ATTENZIONE!:
        i dataset di produzione rinnovabile sono reali ora per ora e ora usano una **media geografica pesata NORD/SUD**.
        Puoi regolare dalla sidebar la quota di NORD per il fotovoltaico e per l'eolico.

        ### 🌿 Modello LCA (Life Cycle Assessment)
        Le emissioni sono calcolate sull'intero ciclo di vita (dati IPCC):
        - **Fotovoltaico:** 45 gCO₂/kWh
        - **Eolico:** 11 gCO₂/kWh
        - **Idroelettrico:** 24 gCO₂/kWh
        - **Nucleare:** 12 gCO₂/kWh
        - **Batterie:** 50 gCO₂/kWh (per energia erogata)
        - **Gas Naturale:** 550 gCO₂/kWh
        """)

    col_vuota, col_bottone = st.columns([4, 1])
    with col_bottone:
        if st.button("ℹ️ Info / Istruzioni / Fonti"):
            mostra_spiegazione()

    st.sidebar.header("🗺️ Mix geografico delle curve")
    quota_eolico_nord_pct = st.sidebar.slider(
        "% eolico NORD", min_value=0.0, max_value=100.0, value=round(DEFAULT_WIND_NORD_SHARE * 100, 2), step=0.1,
        help="La quota SUD è calcolata automaticamente come 100 - quota NORD."
    )
    quota_fotovoltaico_nord_pct = st.sidebar.slider(
        "% fotovoltaico NORD", min_value=0.0, max_value=100.0, value=round(DEFAULT_PV_NORD_SHARE * 100, 2), step=0.1,
        help="La quota SUD è calcolata automaticamente come 100 - quota NORD."
    )
    st.sidebar.caption(
        f"Default realistici: FV NORD {DEFAULT_PV_NORD_SHARE * 100:.2f}% | FV SUD {(1 - DEFAULT_PV_NORD_SHARE) * 100:.2f}% | "
        f"Eolico NORD {DEFAULT_WIND_NORD_SHARE * 100:.2f}% | Eolico SUD {(1 - DEFAULT_WIND_NORD_SHARE) * 100:.2f}%"
    )

    st.sidebar.header("⚙️ Parametri di Mercato")
    mercato = {
        'cfd_pv': st.sidebar.slider("CfD Fotovoltaico (€/MWh)", 20.0, 150.0, 60.0, step=5.0),
        'cfd_wind': st.sidebar.slider("CfD Eolico (€/MWh)", 30.0, 150.0, 80.0, step=5.0),
        'cfd_nuc': st.sidebar.slider("CfD Nucleare (€/MWh)", 50.0, 200.0, 120.0, step=5.0),
        'bess_capex': st.sidebar.slider("CAPEX Batterie (€/MWh installato)", 50000.0, 300000.0, 100000.0, step=10000.0),
        'wacc_bess': st.sidebar.slider("WACC Batterie (%)", 0.0, 15.0, 5.0, step=0.5) / 100,
        'bess_opex_fix': st.sidebar.slider("Manutenzione Annua BESS (% del CAPEX)", 0.0, 5.0, 1.5, step=0.1) / 100,
        'bess_vita': 15,
        'gas_eur_mwh': st.sidebar.slider("Prezzo Gas / Fossili (€/MWh)", 30.0, 300.0, 130.0, step=10.0),
        'costo_base_integrazione': st.sidebar.slider(
            "Costo Integrazione Rete (€/MWh)", 0.0, 20.0, 10.0,
            help="Costo extra per bilanciamento e rete per gestire fotovoltaico ed eolico."
        ),
        'voll': 3000.0
    }

    cartella_script = os.path.dirname(os.path.abspath(__file__))
    file_fotovoltaico = os.path.join(cartella_script, "dataset_fotovoltaico_produzione.csv")
    file_gme = os.path.join(cartella_script, "gme.xlsx")
    file_eolico = os.path.join(cartella_script, "dataset_eolico_produzione.csv")

    quota_fotovoltaico_nord = quota_fotovoltaico_nord_pct / 100.0
    quota_eolico_nord = quota_eolico_nord_pct / 100.0

    df_completo, usato_fallback = carica_dati_v2(
        file_fotovoltaico,
        file_gme,
        file_eolico,
        quota_fotovoltaico_nord,
        quota_eolico_nord,
    )
    
    if usato_fallback:
        st.warning("⚠️ Impossibile estrarre i consumi dal file GME fornito. È stato attivato il 'Carico di Emergenza' (Profilo Italiano Standard) per permetterti di utilizzare comunque la dashboard.")

    with st.spinner("Calcolo della rete elettrica... (Solo al primo avvio o quando cambia la geografia delle curve)"):
        risultati_fisici = simula_tutti_scenari_fisici(
            df_completo['Fattore_Capacita_PV'].to_numpy(dtype=np.float64),
            df_completo['Fattore_Capacita_Wind'].to_numpy(dtype=np.float64),
            df_completo['Fabbisogno_MW'].to_numpy(dtype=np.float64),
        )

    miglior_config, df_plot = applica_economia_e_trova_ottimo(risultati_fisici, df_completo, mercato)

    st.subheader("🏆 Il Miglior Compromesso (Ottimo Economico + 5% Tolleranza Green)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Costo Bolletta", f"{miglior_config['Costo_Bolletta']:.1f} €/MWh")
    col2.metric("Carbon Intensity (LCA)", f"{miglior_config['Carbon_Intensity']:.1f} gCO₂/kWh")
    col3.metric("Nucleare Richiesto", f"{miglior_config['Nuc_GW']} GW")
    col4.metric("Batterie Richieste", f"{miglior_config['BESS_GWh']} GWh")

    st.markdown(
        f"**Mix Impianti:** {miglior_config['PV_GW']} GW Solare | {miglior_config['Wind_GW']} GW Eolico | "
        f"**Spreco Rete:** {miglior_config['Overgen_TWh']:.1f} TWh/anno"
    )
    st.caption(
        f"Curve usate nel calcolo: FV NORD {quota_fotovoltaico_nord_pct:.2f}% / SUD {100 - quota_fotovoltaico_nord_pct:.2f}% | "
        f"Eolico NORD {quota_eolico_nord_pct:.2f}% / SUD {100 - quota_eolico_nord_pct:.2f}%"
    )

    st.subheader("📊 Frontiera di Pareto: Costi vs Emissioni (Interattivo!)")

    fig = px.scatter(
        df_plot,
        x='Carbon_Intensity',
        y='Costo_Bolletta',
        color='Nuc_GW',
        color_continuous_scale='Plasma',
        hover_data=['PV_GW', 'Wind_GW', 'BESS_GWh'],
        labels={
            'Carbon_Intensity': "Carbon Intensity Media LCA (gCO₂/kWh)",
            'Costo_Bolletta': "Costo Medio in Bolletta (€/MWh)",
            'Nuc_GW': "Nucleare (GW)"
        }
    )

    fig.add_trace(go.Scatter(
        x=[miglior_config['Carbon_Intensity']],
        y=[miglior_config['Costo_Bolletta']],
        mode='markers',
        marker=dict(color='lime', size=15, line=dict(color='black', width=2)),
        name='Ottimo Scelto',
        hoverinfo='skip'
    ))

    fig.update_layout(xaxis_autorange="reversed", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 5. TRAIETTORIA DI TRANSIZIONE E COSTO DEL RITARDO
    # ==========================================
    st.markdown("---")
    st.subheader("🛤️ Traiettoria Reale: Il costo dell'attesa")
    st.markdown("I ritardi burocratici o i lunghi tempi di costruzione hanno un costo occulto: **nel frattempo si brucia gas**. Modifica i tempi di cantiere per vedere quanti miliardi ci costa una transizione lenta.")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        anni_transizione = st.slider("Orizzonte di transizione (Anni)", 10, 40, 20)
    
    with st.expander("⏱️ Personalizza i tempi di deploy (Inizio -> Fine Lavori)"):
        st.caption("L'anno 'Inizio' è quando il primo GW entra in rete. L'anno 'Fine' è quando si raggiunge il target del Mix Ottimo.")
        c1, c2, c3, c4 = st.columns(4)
        pv_start = c1.number_input("Inizio PV (Anno)", 0, 40, 1)
        pv_end = c1.number_input("Fine PV (Anno)", 1, 40, 15)
        
        wind_start = c2.number_input("Inizio Eolico", 0, 40, 3)
        wind_end = c2.number_input("Fine Eolico", 1, 40, 18)
        
        bess_start = c3.number_input("Inizio BESS", 0, 40, 1)
        bess_end = c3.number_input("Fine BESS", 1, 40, 15)
        
        nuc_start = c4.number_input("Inizio Nucleare", 0, 40, 12)
        nuc_end = c4.number_input("Fine Nucleare", 1, 50, 20)

    status_quo = df_plot.sort_values(by=['PV_GW', 'Wind_GW', 'BESS_GWh', 'Nuc_GW']).iloc[0]

    array_pv = df_completo['Fattore_Capacita_PV'].to_numpy(dtype=np.float64)
    array_wind = df_completo['Fattore_Capacita_Wind'].to_numpy(dtype=np.float64)
    array_fabbisogno = df_completo['Fabbisogno_MW'].to_numpy(dtype=np.float64)
    
    def calcola_capacita_anno(anno, start_yr, end_yr, val_start, val_target, step_wise=False):
        if end_yr <= start_yr: 
            end_yr = start_yr + 1 
            
        if anno <= start_yr:
            return val_start
        elif anno >= end_yr:
            return val_target
        else:
            quota = (anno - start_yr) / (end_yr - start_yr)
            valore = val_start + (val_target - val_start) * quota
            if step_wise:
                return np.floor(valore)
            return valore

    storia_transizione = []
    costo_gas_cumulato_mld = 0.0

    for anno in range(anni_transizione + 1):
        pv_gw = calcola_capacita_anno(anno, pv_start, pv_end, status_quo['PV_GW'], miglior_config['PV_GW'])
        wind_gw = calcola_capacita_anno(anno, wind_start, wind_end, status_quo['Wind_GW'], miglior_config['Wind_GW'])
        bess_gwh = calcola_capacita_anno(anno, bess_start, bess_end, status_quo['BESS_GWh'], miglior_config['BESS_GWh'])
        nuc_gw = calcola_capacita_anno(anno, nuc_start, nuc_end, status_quo['Nuc_GW'], miglior_config['Nuc_GW'], step_wise=True)
        
        gas_mwh, def_mwh, over_mwh, _, _ = simula_rete_light_fast(
            array_pv, array_wind, array_fabbisogno,
            pv_gw * 1000.0, wind_gw * 1000.0, nuc_gw * 1000.0, bess_gwh * 1000.0, 
            50000.0, 50000.0, 2500.0, 12000.0, 5000000.0, 2850.0
        )
        
        costo_gas_anno_mld = (gas_mwh * mercato['gas_eur_mwh']) / 1e9
        costo_gas_cumulato_mld += costo_gas_anno_mld
        
        storia_transizione.append({
            'Anno': anno,
            'PV_GW': pv_gw,
            'Wind_GW': wind_gw,
            'Nuc_GW': nuc_gw,
            'BESS_GWh': bess_gwh,
            'Gas_TWh': gas_mwh / 1e6,
            'Costo_Gas_Mld': costo_gas_anno_mld,
            'Deficit_TWh': def_mwh / 1e6
        })

    df_t = pd.DataFrame(storia_transizione)

    st.error(f"💸 **Spesa Cumulata per il Gas durante la transizione:** {costo_gas_cumulato_mld:.1f} Miliardi di €")
    st.caption(f"Questo è il costo generato dal bruciare gas negli anni intermedi, prima che tutti i nuovi impianti siano a regime. Se ritardi le autorizzazioni, questa cifra esplode.")

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['PV_GW'], mode='lines', stackgroup='one', name='Fotovoltaico (GW)', fillcolor='gold', line=dict(width=0.5)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Wind_GW'], mode='lines', stackgroup='one', name='Eolico (GW)', fillcolor='lightskyblue', line=dict(width=0.5)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Nuc_GW'], mode='lines', stackgroup='one', name='Nucleare (GW)', fillcolor='mediumpurple', line=dict(width=1.5, shape='hv')), secondary_y=False)
    
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Gas_TWh'], mode='lines+markers', name='Consumo Gas (TWh)', line=dict(color='red', width=4), marker=dict(size=6)), secondary_y=True)

    fig2.update_layout(
        title="Dinamica di Costruzione e Riduzione del Gas",
        xaxis_title="Anno di Transizione (0 = Oggi)",
        hovermode="x unified",
        height=500
    )
    fig2.update_yaxes(title_text="Capacità Installata (<b>GW</b>)", secondary_y=False)
    fig2.update_yaxes(title_text="Gas Bruciato (<b>TWh/anno</b>)", secondary_y=True, range=[0, df_t['Gas_TWh'].max() * 1.1])

    st.plotly_chart(fig2, use_container_width=True)
    
    deficit_max = df_t['Deficit_TWh'].max()
    if deficit_max > 0.5:
        st.warning(f"⚠️ Attenzione: Durante la transizione, la mancanza di impianti pronti causa un picco di deficit (blackout) di **{deficit_max:.1f} TWh**. Valuta di accelerare le batterie o mantenere più gas di riserva.")


    # ==========================================
    # 6. MODULO POWER-TO-GAS (CO-OTTIMIZZAZIONE ORARIA VRE+ELC+BATT+GRID)
    # ==========================================
    st.markdown("---")
    st.header("🏭 L'Ultimo Miglio: Power-to-Gas con Co-Ottimizzazione di Rete")
    st.markdown("Il motore Numba simula **6.000 combinazioni** di impianti (Pannelli + Elettrolizzatori + Batterie) per 8760 ore, incrociandole con la curva di scarto della rete. Troverà la combinazione che **minimizza il costo totale (LCOH)** sfruttando l'energia gratuita estiva della rete per abbassare il CAPEX dell'impianto verde.")

    hc1, hc2, hc3, hc4 = st.columns(4)
    eff_meth = hc1.slider("Efficienza Metanazione (H₂ -> CH₄) [%]", 70.0, 90.0, 78.0, step=1.0) / 100
    costo_co2 = hc2.slider("Costo CO₂ Biogenica [€/ton]", 0, 150, 70, step=5)
    capex_elc = hc3.slider("CAPEX Elettrolizzatore [€/kW]", 500, 2000, 1000, step=100)
    capex_meth = hc4.slider("CAPEX Metanatore [€/kW]", 200, 1000, 480, step=10)

    gas_da_sostituire_twh = miglior_config['gas_mwh'] / 1e6
    energia_el_necessaria_h2_twh = gas_da_sostituire_twh / eff_meth
    co2_necessaria_kton = (gas_da_sostituire_twh * 1e6) * 0.198 / 1000
    costo_fornitura_co2_mln = (co2_necessaria_kton * 1000 * costo_co2) / 1e6 

    if gas_da_sostituire_twh <= 0.1:
        st.success("🎉 Complimenti! La configurazione scelta non usa quasi più gas. Non serve un piano Power-to-Gas massivo.")
    else:
        st.info(f"🎯 **Target:** Per produrre {gas_da_sostituire_twh:.1f} TWh di Metano Sintetico servono **{energia_el_necessaria_h2_twh:.1f} TWh di Idrogeno elettrico** e **{co2_necessaria_kton:.1f} kton di CO₂ biogenica**.")

        with st.spinner("Analisi di 70 milioni di scenari orari (Numba in esecuzione)..."):
            
            # Estraiamo la curva oraria del curtailment di rete (la vera e propria disponibilità oraria)
            overgen_orario = estrai_curva_overgen_oraria(
                array_pv, array_wind, array_fabbisogno,
                miglior_config['PV_GW'] * 1000.0, miglior_config['Wind_GW'] * 1000.0,
                miglior_config['Nuc_GW'] * 1000.0, miglior_config['BESS_GWh'] * 1000.0,
                50000.0, 2500.0, 5000000.0, 2850.0
            )

            wacc = mercato['wacc_bess']
            vita_impianti = 20
            crf = (wacc * (1 + wacc)**vita_impianti) / ((1 + wacc)**vita_impianti - 1) if wacc > 0 else 1/vita_impianti
            
            h2_target_mwh_el = energia_el_necessaria_h2_twh * 1e6

            # === OPZIONE 1: VIA NUCLEARE ===
            # Ottimizza solo la taglia dell'elettrolizzatore, alimentato dal Reattore + Rete.
            cf_nuc_impianto = 0.92
            taglia_elc_nuc_gw, free_nuc_mwh = ottimizza_h2_nucleare(
                overgen_orario, h2_target_mwh_el, capex_elc, mercato['cfd_nuc'], cf_nuc_impianto, crf
            )
            
            energia_acquistata_nuc_mwh = h2_target_mwh_el - free_nuc_mwh
            costo_energia_nuc_mln = (energia_acquistata_nuc_mwh * mercato['cfd_nuc']) / 1e6
            capex_tot_elc_nuc_mln = taglia_elc_nuc_gw * capex_elc
            curtailment_recuperato_nuc_twh = free_nuc_mwh / 1e6


            # === OPZIONE 2: VIA RINNOVABILE (CO-OTTIMIZZAZIONE A 3 VARIABILI) ===
            # Costruiamo il profilo della rinnovabile dedicata (mix 60% FV, 40% Wind) scalato a 1 GW
            quota_pv = miglior_config['PV_GW'] / (miglior_config['PV_GW'] + miglior_config['Wind_GW'] + 1e-9)
            vre_profile_1GW = (array_pv * quota_pv + array_wind * (1 - quota_pv)) * 1000.0 # MW
            
            # Calcoliamo il costo annualizzato di 1 GW di rinnovabili
            ore_eq_vre_1gw = np.sum(vre_profile_1GW)
            lcoe_vre_medio = (mercato['cfd_pv'] * quota_pv) + (mercato['cfd_wind'] * (1 - quota_pv))
            costo_annuo_vre_gw = (ore_eq_vre_1gw * lcoe_vre_medio)
            
            # 🚀 L'algoritmo Numba esplora lo spazio 3D alla ricerca del set perfetto
            taglia_elc_vre_gw, taglia_vre_gw, taglia_batt_gwh, free_vre_mwh = co_ottimizza_h2_rinnovabile(
                overgen_orario, vre_profile_1GW, h2_target_mwh_el,
                capex_elc, mercato['bess_capex'], costo_annuo_vre_gw, crf
            )
            
            # Ricalcoliamo le finanze della configurazione vincente
            costo_energia_vre_mln = (taglia_vre_gw * costo_annuo_vre_gw) / 1e6
            capex_tot_elc_vre_mln = taglia_elc_vre_gw * capex_elc
            capex_tot_batt_vre_mln = taglia_batt_gwh * 1000 * mercato['bess_capex'] * crf / 1e6
            curtailment_recuperato_vre_twh = free_vre_mwh / 1e6

        # COSTI METANATORE E BUFFER CO2 (Comuni per le due vie)
        taglia_meth_gw = (gas_da_sostituire_twh * 1000) / (8760 * 0.90)
        capex_tot_meth_mln = taglia_meth_gw * capex_meth
        opex_fix_meth_mln = capex_tot_meth_mln * 0.035
        opex_var_meth_mln = (gas_da_sostituire_twh * 1e6) * 5.44 / 1e6 
        
        co2_buffer_kton = co2_necessaria_kton * (3 / 365)
        capex_co2_storage_mln = (co2_buffer_kton * 1000 * 2528) / 1e6
        costi_meth_comuni_mln_anno = (capex_tot_meth_mln * crf) + opex_fix_meth_mln + opex_var_meth_mln + (capex_co2_storage_mln * crf) + costo_fornitura_co2_mln

        # CALCOLO LCO_CH4 FINALE
        costo_annuo_nuc_mln = (capex_tot_elc_nuc_mln * crf) + costo_energia_nuc_mln + costi_meth_comuni_mln_anno
        costo_annuo_vre_mln = (capex_tot_elc_vre_mln * crf) + capex_tot_batt_vre_mln + costo_energia_vre_mln + costi_meth_comuni_mln_anno
        
        lco_ch4_nuc = (costo_annuo_nuc_mln * 1e6) / (gas_da_sostituire_twh * 1e6)
        lco_ch4_vre = (costo_annuo_vre_mln * 1e6) / (gas_da_sostituire_twh * 1e6)

        # VISUALIZZAZIONE RISULTATI
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("### ⚛️ Via Nucleare (e-CH₄ Rosa)")
            st.markdown(f"- **Taglia Elettrolizzatore:** {taglia_elc_nuc_gw:.1f} GW")
            st.markdown(f"- **Spreco di Rete Recuperato:** {curtailment_recuperato_nuc_twh:.1f} TWh")
            
            diff_nuc = lco_ch4_nuc - mercato['gas_eur_mwh']
            st.metric("Costo Gas Sintetico (LCO_CH₄)", f"{lco_ch4_nuc:.1f} €/MWh", delta=f"{diff_nuc:+.1f} €/MWh vs Gas Fossile", delta_color="inverse")
            
        with col_res2:
            st.markdown("### 🌬️☀️ Via Rinnovabili (e-CH₄ Verde)")
            st.markdown(f"- **Impianto Co-Ottimizzato:** {taglia_vre_gw:.1f} GW Rinnovabili + {taglia_elc_vre_gw:.1f} GW Elettrolizzatore + {taglia_batt_gwh:.1f} GWh Batterie")
            st.markdown(f"- **Spreco di Rete Recuperato:** {curtailment_recuperato_vre_twh:.1f} TWh")
            
            diff_vre = lco_ch4_vre - mercato['gas_eur_mwh']
            st.metric("Costo Gas Sintetico (LCO_CH₄)", f"{lco_ch4_vre:.1f} €/MWh", delta=f"{diff_vre:+.1f} €/MWh vs Gas Fossile", delta_color="inverse")

        df_ptg_costi = pd.DataFrame({
            'Voce': ['CAPEX Elettrolizzatore', 'CAPEX Batterie Dedicate', 'Energia (Rinnovabile Dedicata o Rete)', 'CAPEX Metanatore + CO2', 'OPEX Metanatore + Fornitura CO2'],
            'Nucleare (Milioni €/anno)': [capex_tot_elc_nuc_mln * crf, 0, costo_energia_nuc_mln, (capex_tot_meth_mln + capex_co2_storage_mln) * crf, opex_fix_meth_mln + opex_var_meth_mln + costo_fornitura_co2_mln],
            'Rinnovabili (Milioni €/anno)': [capex_tot_elc_vre_mln * crf, capex_tot_batt_vre_mln, costo_energia_vre_mln, (capex_tot_meth_mln + capex_co2_storage_mln) * crf, opex_fix_meth_mln + opex_var_meth_mln + costo_fornitura_co2_mln]
        })
        df_ptg_melted = df_ptg_costi.melt(id_vars='Voce', var_name='Scenario', value_name='Milioni €/anno')
        
        fig_ptg = px.bar(df_ptg_melted, x='Scenario', y='Milioni €/anno', color='Voce', barmode='stack',
                        title="Scomposizione Costi LCO_CH₄ (Impianto Dedicato + Parassitismo di Rete)",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_ptg, use_container_width=True)

        costo_minimo_ptg_mln = min(costo_annuo_nuc_mln, costo_annuo_vre_mln)
        impatto_reale_sistema = (costo_minimo_ptg_mln * 1e6) / df_completo['Fabbisogno_MW'].sum()
        
        st.error(f"⚡ **Impatto Definitivo sul Sistema:** Produrre il metano sintetico verde/rosa costerà circa **{costo_minimo_ptg_mln/1000:.2f} Miliardi di € all'anno**. Questo aggiungerà **{impatto_reale_sistema:.2f} €/MWh** alla bolletta media nazionale calcolata sopra.")

# ==========================================
# GESTIONE ERRORI
# ==========================================
except FileNotFoundError:
    st.error("⚠️ File dati non trovati! Assicurati che i dataset siano nella stessa cartella dell'app.")
except ValueError as e:
    st.error(f"⚠️ Dati anomali o formattazione errata: {e}")
except KeyError as e:
    st.error(f"⚠️ Errore di lettura campi. Ricarica la pagina. Errore originale: {e}")
except Exception as e:
    st.error(f"⚠️ Errore imprevisto durante l'elaborazione dei dati: {e}")
