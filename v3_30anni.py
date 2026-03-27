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
# FUNZIONI DI SUPPORTO DATI
# ==========================================
def _serie_pesata(df, pesi_colonne, scala=1.0, clip_upper=1.0):
    colonne_mancanti = [col for col in pesi_colonne if col not in df.columns]
    if colonne_mancanti:
        raise KeyError("Nel dataset mancano le colonne: " + ", ".join(colonne_mancanti))
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
    colonna_volumi = df_gme.columns[2]
    if df_gme[colonna_volumi].dtype == 'object':
        df_gme[colonna_volumi] = df_gme[colonna_volumi].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df_gme[colonna_volumi] = pd.to_numeric(df_gme[colonna_volumi], errors='coerce')
    df_gme['Ora'] = pd.to_numeric(df_gme['Ora'], errors='coerce')
    data_convertita = pd.to_datetime(df_gme['Data'], dayfirst=True, errors='coerce')
    ore_aggiuntive = pd.to_timedelta(df_gme['Ora'] - 1, unit='h')
    df_gme['Datetime'] = data_convertita + ore_aggiuntive
    df_gme = df_gme.dropna(subset=['Datetime', colonna_volumi]).copy()
    df_gme.set_index('Datetime', inplace=True)
    df_gme.rename(columns={colonna_volumi: 'Fabbisogno_MW'}, inplace=True)
    return df_gme[['Fabbisogno_MW']]

@st.cache_data
def carica_profili_rinnovabili(file_fotovoltaico, file_eolico):
    df_pv = pd.read_csv(file_fotovoltaico)
    df_pv['time'] = pd.to_datetime(df_pv['time'], errors='coerce')
    df_pv.set_index('time', inplace=True)
    df_wind = pd.read_csv(file_eolico)
    df_wind['time'] = pd.to_datetime(df_wind['time'], errors='coerce')
    df_wind.set_index('time', inplace=True)
    profili = {
        'pv_nord': pd.Series(_serie_pesata(df_pv, PV_WEIGHTS_NORD, scala=1000.0).values, index=df_pv.index),
        'pv_sud': pd.Series(_serie_pesata(df_pv, PV_WEIGHTS_SUD, scala=1000.0).values, index=df_pv.index),
        'wind_nord': pd.Series(_serie_pesata(df_wind, WIND_WEIGHTS_NORD, scala=1.0).values, index=df_wind.index),
        'wind_sud': pd.Series(_serie_pesata(df_wind, WIND_WEIGHTS_SUD, scala=1.0).values, index=df_wind.index),
    }
    return profili

@st.cache_data
def carica_dati(file_fotovoltaico, file_gme, file_eolico, quota_pv_nord, quota_eolico_nord):
    df_gme = leggi_gme(file_gme)
    profili = carica_profili_rinnovabili(file_fotovoltaico, file_eolico)
    profilo_pv = (profili['pv_nord'] * quota_pv_nord) + (profili['pv_sud'] * (1.0 - quota_pv_nord))
    profilo_wind = (profili['wind_nord'] * quota_eolico_nord) + (profili['wind_sud'] * (1.0 - quota_eolico_nord))
    df_completo = df_gme.copy()
    df_completo['Fattore_Capacita_PV'] = _mappa_profilo_annuale_su_indice(profilo_pv, df_completo.index)
    df_completo['Fattore_Capacita_Wind'] = _mappa_profilo_annuale_su_indice(profilo_wind, df_completo.index)
    return df_completo.ffill()

# ==========================================
# 2. SIMULAZIONE FISICA E CAPACITÀ (Numba)
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
    potenza_nucleare_costante = nucleare_mw
    gas_usato_totale, deficit_totale, overgen_totale = 0.0, 0.0, 0.0
    hydro_dispatched_totale, bess_scarica_totale = 0.0, 0.0
    sqrt_eff = np.sqrt(efficienza_bess)

    for t in range(ore):
        soc_hydro += hydro_inflow_mw
        if soc_hydro > hydro_bacino_max_mwh: soc_hydro = hydro_bacino_max_mwh
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

@njit
def calcola_capacita_anno_rate(anno, start_yr, val_start, val_target, rate, step_wise=False):
    if anno <= start_yr: 
        return val_start
    anni_attivi = anno - start_yr
    valore = val_start + (anni_attivi * rate)
    if valore > val_target:
        valore = val_target
    if step_wise: 
        return np.floor(valore)
    return valore

@njit
def simula_scenario_30_anni(prod_pv, prod_wind, fabbisogno, 
                            pv_target, wind_target, nuc_target, bess_target,
                            pv_sq, wind_sq, nuc_sq, bess_sq,
                            t_start_numba, rate_numba, anni_transizione=30):
    gas_tot, def_tot, over_tot, hydro_disp_tot, bess_out_tot = 0.0, 0.0, 0.0, 0.0, 0.0
    pv_gen_tot, wind_gen_tot, nuc_gen_tot = 0.0, 0.0, 0.0
    bess_installed_tot_mwh_years = 0.0
    vre_gen_tot = 0.0 
    
    ore_anno = len(fabbisogno)
    
    for anno in range(anni_transizione + 1):
        pv_gw = calcola_capacita_anno_rate(anno, t_start_numba['pv'], pv_sq, pv_target, rate_numba['pv'])
        wind_gw = calcola_capacita_anno_rate(anno, t_start_numba['wind'], wind_sq, wind_target, rate_numba['wind'])
        nuc_gw = calcola_capacita_anno_rate(anno, t_start_numba['nuc'], nuc_sq, nuc_target, rate_numba['nuc'], step_wise=True)
        bess_gwh = calcola_capacita_anno_rate(anno, t_start_numba['bess'], bess_sq, bess_target, rate_numba['bess'])
        
        # --- LIMITE C-RATE 0.5 ---
        bess_mw = bess_gwh * 1000.0 * 0.5

        gas, dfc, ovr, hyd, bss = simula_rete_light_fast(
            prod_pv, prod_wind, fabbisogno,
            pv_gw * 1000.0, wind_gw * 1000.0, nuc_gw * 1000.0, bess_gwh * 1000.0, 
            bess_mw, 50000.0, 2500.0, 12000.0, 5000000.0, 2850.0
        )
        
        # --- SEZIONE RICUCITA (PRIMA ERA ORFANATA) ---
        gas_tot += gas
        def_tot += dfc
        over_tot += ovr
        hydro_disp_tot += hyd
        bess_out_tot += bss
        bess_installed_tot_mwh_years += (bess_gwh * 1000.0)
            
        pv_gen_anno = np.sum(prod_pv * (pv_gw * 1000.0))
        wind_gen_anno = np.sum(prod_wind * (wind_gw * 1000.0))
        pv_gen_tot += pv_gen_anno
        wind_gen_tot += wind_gen_anno
        nuc_gen_tot += (nuc_gw * 1000.0 * 0.90 * ore_anno)
        vre_gen_tot += (pv_gen_anno + wind_gen_anno)
            
    return (gas_tot, def_tot, over_tot, hydro_disp_tot, bess_out_tot, 
            pv_gen_tot, wind_gen_tot, nuc_gen_tot, bess_installed_tot_mwh_years, vre_gen_tot)

@njit
def simula_rete_dettaglio_orario(produzione_pv, produzione_wind, fabbisogno,
                                 pv_mw, wind_mw, nucleare_mw, bess_mwh, bess_mw, gas_mw,
                                 hydro_fluente_mw, hydro_bacino_mw, hydro_bacino_max_mwh, hydro_inflow_mw,
                                 efficienza_bess=0.9):
    ore = len(fabbisogno)
    soc_corrente = bess_mwh * 0.5
    soc_hydro = hydro_bacino_max_mwh * 0.5
    
    prod_pv_array = produzione_pv * pv_mw
    prod_wind_array = produzione_wind * wind_mw
    potenza_nucleare_costante = nucleare_mw
    sqrt_eff = np.sqrt(efficienza_bess)

    # Array di output per le 8760 ore
    out_pv = np.zeros(ore)
    out_wind = np.zeros(ore)
    out_nuc = np.full(ore, potenza_nucleare_costante)
    out_hydro_fluente = np.full(ore, hydro_fluente_mw)
    out_hydro_bacino = np.zeros(ore)
    out_bess_scarica = np.zeros(ore)
    out_gas = np.zeros(ore)
    out_deficit = np.zeros(ore)

    for t in range(ore):
        soc_hydro += hydro_inflow_mw
        if soc_hydro > hydro_bacino_max_mwh: soc_hydro = hydro_bacino_max_mwh

        out_pv[t] = prod_pv_array[t]
        out_wind[t] = prod_wind_array[t]

        generazione_base = prod_pv_array[t] + prod_wind_array[t] + hydro_fluente_mw + potenza_nucleare_costante
        bilancio_netto = generazione_base - fabbisogno[t]

        if bilancio_netto > 0:
            spazio_libero_batteria = bess_mwh - soc_corrente
            potenza_carica = min(bilancio_netto, bess_mw, spazio_libero_batteria / sqrt_eff)
            soc_corrente += potenza_carica * sqrt_eff
        else:
            energia_richiesta = abs(bilancio_netto)
            potenza_scarica_bess = min(energia_richiesta, bess_mw)
            energia_out_bess = potenza_scarica_bess / sqrt_eff
            
            if soc_corrente >= energia_out_bess:
                soc_corrente -= energia_out_bess
                energia_richiesta -= potenza_scarica_bess
                out_bess_scarica[t] = potenza_scarica_bess
            else:
                energia_disp_bess = soc_corrente * sqrt_eff
                soc_corrente = 0.0
                energia_richiesta -= energia_disp_bess
                out_bess_scarica[t] = energia_disp_bess

            if energia_richiesta > 0:
                potenza_scarica_hydro = min(energia_richiesta, hydro_bacino_mw)
                if soc_hydro >= potenza_scarica_hydro:
                    soc_hydro -= potenza_scarica_hydro
                    energia_richiesta -= potenza_scarica_hydro
                    out_hydro_bacino[t] = potenza_scarica_hydro
                else:
                    out_hydro_bacino[t] = soc_hydro
                    energia_richiesta -= soc_hydro
                    soc_hydro = 0.0

            if energia_richiesta > 0:
                uso_gas = min(energia_richiesta, gas_mw)
                out_gas[t] = uso_gas
                out_deficit[t] = energia_richiesta - uso_gas

    return out_pv, out_wind, out_nuc, out_hydro_fluente, out_hydro_bacino, out_bess_scarica, out_gas, out_deficit

# ==========================================
# 3. HELPER PYTHON E MOTORE SCENARI
# ==========================================
def get_reached_capacity(anno_fine, start_yr, val_start, val_target, rate, step_wise=False):
    if anno_fine <= start_yr:
        return val_start
    valore = val_start + ((anno_fine - start_yr) * rate)
    if valore > val_target:
        valore = val_target
    if step_wise:
        return np.floor(valore)
    return valore

@st.cache_data
def simula_motore_30_anni(array_pv, array_wind, array_fabbisogno, t_start_py, rate_py, anni_transizione=30):
    from numba.core import types
    from numba.typed import Dict

    d_start = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    d_rate = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for k, v in t_start_py.items(): d_start[k] = float(v)
    for k, v in rate_py.items(): d_rate[k] = float(v)

    pv_sq, wind_sq, nuc_sq, bess_sq = 40.0, 10.0, 0.0, 10.0

    def max_reach(sq, limit, t_start, rate):
        active_years = max(0, anni_transizione - t_start)
        return min(limit, sq + (active_years * rate))

    max_pv = max_reach(pv_sq, 150.0, t_start_py['pv'], rate_py['pv'])
    max_wind = max_reach(wind_sq, 90.0, t_start_py['wind'], rate_py['wind'])
    max_bess = max_reach(bess_sq, 300.0, t_start_py['bess'], rate_py['bess'])
    max_nuc = np.floor(max_reach(nuc_sq, 20.0, t_start_py['nuc'], rate_py['nuc']))

    def get_valid_targets(sq, max_val, base_targets):
        valid = [float(sq)]
        for t in base_targets:
            if sq < t < max_val:
                valid.append(float(t))
        if max_val > sq:
            valid.append(float(max_val))
        return sorted(list(set([round(v, 1) for v in valid])))

    scenari_pv_gw = get_valid_targets(pv_sq, max_pv, [70, 100, 150,100])
    scenari_wind_gw = get_valid_targets(wind_sq, max_wind, [30, 60, 90,120])
    scenari_bess_gwh = get_valid_targets(bess_sq, max_bess, [50,100, 150, 300])
    scenari_nuc_gw = get_valid_targets(nuc_sq, max_nuc, [5, 10,15, 20])

    risultati_30y = []

    for pv in scenari_pv_gw:
        for wind in scenari_wind_gw:
            for bess in scenari_bess_gwh:
                for nuc in scenari_nuc_gw:
                    
                    r_pv = get_reached_capacity(anni_transizione, t_start_py['pv'], pv_sq, float(pv), rate_py['pv'])
                    r_wind = get_reached_capacity(anni_transizione, t_start_py['wind'], wind_sq, float(wind), rate_py['wind'])
                    r_nuc = get_reached_capacity(anni_transizione, t_start_py['nuc'], nuc_sq, float(nuc), rate_py['nuc'], True)
                    r_bess = get_reached_capacity(anni_transizione, t_start_py['bess'], bess_sq, float(bess), rate_py['bess'])

                    (gas_tot, def_tot, over_tot, hydro_disp_tot, bess_out_tot, 
                     pv_gen_tot, wind_gen_tot, nuc_gen_tot, bess_inst_years, vre_tot) = simula_scenario_30_anni(
                        array_pv, array_wind, array_fabbisogno,
                        float(pv), float(wind), float(nuc), float(bess),
                        pv_sq, wind_sq, nuc_sq, bess_sq,
                        d_start, d_rate, anni_transizione
                    )
                    
                    risultati_30y.append({
                        'Target_PV': pv, 'Target_Wind': wind, 'Target_BESS': bess, 'Target_Nuc': nuc,
                        'Reached_PV': r_pv, 'Reached_Wind': r_wind, 'Reached_BESS': r_bess, 'Reached_Nuc': r_nuc,
                        'gas_mwh': gas_tot, 'deficit_mwh': def_tot, 'overgen_mwh': over_tot,
                        'hydro_disp_mwh': hydro_disp_tot, 'bess_scarica_mwh': bess_out_tot,
                        'pv_gen_mwh': pv_gen_tot, 'wind_gen_mwh': wind_gen_tot, 'nuc_gen_mwh': nuc_gen_tot,
                        'bess_inst_years': bess_inst_years, 'vre_gen_tot': vre_tot
                    })

    return risultati_30y

def applica_economia_cumulata(risultati_30y, fabbisogno_annuo_mwh, mercato, anni_transizione=30):
    fabbisogno_cumulato = fabbisogno_annuo_mwh * (anni_transizione + 1)
    hydro_fluente_cumulato = 2500.0 * 8760 * (anni_transizione + 1)
    LCA_EMISSIONI = {'pv': 45.0, 'wind': 11.0, 'hydro': 24.0, 'nuc': 12.0, 'bess': 50.0, 'gas': 550.0}
    
    wacc = mercato.get('wacc_bess', 0.05)
    vita = mercato.get('bess_vita', 15)
    opex_f_rate = mercato.get('bess_opex_fix', 0.015)
    crf = (wacc * (1 + wacc) ** vita) / ((1 + wacc) ** vita - 1) if wacc > 0 else 1 / vita
    costo_bess_annuo_per_mwh = (mercato['bess_capex'] * crf) + (mercato['bess_capex'] * opex_f_rate)

    storia = []
    for r in risultati_30y:
        costo_pv_tot = r['pv_gen_mwh'] * mercato['cfd_pv']
        costo_wind_tot = r['wind_gen_mwh'] * mercato['cfd_wind']
        costo_nuc_tot = r['nuc_gen_mwh'] * mercato['cfd_nuc']
        costo_hydro_tot = (hydro_fluente_cumulato + r['hydro_disp_mwh']) * mercato['gas_eur_mwh']
        costo_gas_tot = r['gas_mwh'] * mercato['gas_eur_mwh']
        
        # --- OVRSIZING BATTERIE (+20%) ---
        bess_nominale_acquistato = r['bess_inst_years'] * 1.20
        costo_bess_tot = bess_nominale_acquistato * costo_bess_annuo_per_mwh

        costo_blackout_tot = r['deficit_mwh'] * mercato['voll']
        
        quota_vre_media = r['vre_gen_tot'] / fabbisogno_cumulato
        costo_base_integr = mercato['costo_base_integrazione'] * (quota_vre_media ** 2)
        costo_sistema_totale = r['vre_gen_tot'] * costo_base_integr
        
        costo_totale_30y = (costo_pv_tot + costo_wind_tot + costo_nuc_tot + costo_hydro_tot + 
                            costo_gas_tot + costo_bess_tot + costo_blackout_tot + costo_sistema_totale)
        
        costo_medio_bolletta = costo_totale_30y / fabbisogno_cumulato
        percentuale_gas = (r['gas_mwh'] / fabbisogno_cumulato) * 100
        
        emi_tot = (r['pv_gen_mwh'] * LCA_EMISSIONI['pv'] +
                   r['wind_gen_mwh'] * LCA_EMISSIONI['wind'] +
                   (hydro_fluente_cumulato + r['hydro_disp_mwh']) * LCA_EMISSIONI['hydro'] +
                   r['nuc_gen_mwh'] * LCA_EMISSIONI['nuc'] +
                   r['bess_scarica_mwh'] * LCA_EMISSIONI['bess'] +
                   r['gas_mwh'] * LCA_EMISSIONI['gas'])
        
        cfd_medio_vre = (mercato['cfd_pv'] + mercato['cfd_wind']) / 2
        valore_spreco_eur = r['overgen_mwh'] * cfd_medio_vre

        storia.append({
            'Configurazione': f"{r['Reached_PV']}P|{r['Reached_Wind']}W|{r['Reached_BESS']}B|{r['Reached_Nuc']}N",
            'Target_PV': r['Target_PV'], 'Target_Wind': r['Target_Wind'], 'Target_BESS': r['Target_BESS'], 'Target_Nuc': r['Target_Nuc'],
            'Reached_PV': r['Reached_PV'], 'Reached_Wind': r['Reached_Wind'], 'Reached_BESS': r['Reached_BESS'], 'Reached_Nuc': r['Reached_Nuc'],
            'Costo_Medio_30y': costo_medio_bolletta,
            'Spesa_Totale_Mld_30y': costo_totale_30y / 1e9,
            'Carbon_Intensity_30y': emi_tot / fabbisogno_cumulato,
            'Gas_%_30y': percentuale_gas,
            'Overgen_TWh_30y': r['overgen_mwh'] / 1e6,
            'Gas_Mld_30y': costo_gas_tot / 1e9,
            'Valore_Spreco_Mld_30y': valore_spreco_eur / 1e9
        })

    df_risultati = pd.DataFrame(storia)

    caso_0 = df_risultati[
        (df_risultati['Target_PV'] == 40.0) & 
        (df_risultati['Target_Wind'] == 10.0) & 
        (df_risultati['Target_BESS'] == 10.0) & 
        (df_risultati['Target_Nuc'] == 0.0)
    ]
    costo_caso_0 = caso_0.iloc[0]['Spesa_Totale_Mld_30y'] if not caso_0.empty else df_risultati['Spesa_Totale_Mld_30y'].max()
    df_risultati['Risparmio_Mld_30y'] = costo_caso_0 - df_risultati['Spesa_Totale_Mld_30y']

    df_risultati['Sogni_Infranti'] = (
        (df_risultati['Target_PV'] - df_risultati['Reached_PV']) +
        (df_risultati['Target_Wind'] - df_risultati['Reached_Wind']) +
        (df_risultati['Target_BESS'] - df_risultati['Reached_BESS']) +
        (df_risultati['Target_Nuc'] - df_risultati['Reached_Nuc'])
    )
    df_risultati = df_risultati.sort_values('Sogni_Infranti').drop_duplicates(
        subset=['Reached_PV', 'Reached_Wind', 'Reached_BESS', 'Reached_Nuc']
    ).drop(columns=['Sogni_Infranti'])

    min_c = df_risultati['Costo_Medio_30y'].min()
    max_c = df_risultati['Costo_Medio_30y'].max()
    min_e = df_risultati['Carbon_Intensity_30y'].min()
    max_e = df_risultati['Carbon_Intensity_30y'].max()

    if max_c == min_c: max_c = min_c + 1
    if max_e == min_e: max_e = min_e + 1

    df_risultati['Norm_Cost'] = (df_risultati['Costo_Medio_30y'] - min_c) / (max_c - min_c)
    df_risultati['Norm_Emi'] = (df_risultati['Carbon_Intensity_30y'] - min_e) / (max_e - min_e)
    df_risultati['Distanza_Utopia'] = np.sqrt(df_risultati['Norm_Cost']**2 + df_risultati['Norm_Emi']**2)

    miglior_config = df_risultati.sort_values(by='Distanza_Utopia').iloc[0].to_dict()
    df_risultati = df_risultati.drop(columns=['Norm_Cost', 'Norm_Emi', 'Distanza_Utopia'])
    
    return miglior_config, df_risultati

# ==========================================
# 4. INTERFACCIA UTENTE (STREAMLIT)
# ==========================================
st.title("⚡ QUANTO RISPARMIEREMMO CON LA TRANSIZIONE?")
st.markdown("Valuta le emissioni e i costi sull'**intero arco della transizione**. Imposta quanto velocemente riesci a installare nuova capacità ogni anno.")

# --- SIDEBAR: TIMING E CAPACITÀ INSTALLATIVA ---
st.sidebar.header("⏱️ Velocità nelle installazioni")
st.sidebar.caption("Definisce quando inizia la costruzione (0 = Oggi) e quanti GW/GWh all'anno riesci ad aggiungere alla rete.")
anni_transizione = st.sidebar.slider("Orizzonte di transizione (Anni)", 10, 40, 30)

col_t1, col_t2 = st.sidebar.columns(2)
t_start = {
    'pv': col_t1.number_input("Inizio PV (Anno)", 0, 40, 1),
    'wind': col_t1.number_input("Inizio Eol", 0, 40, 3),
    'bess': col_t1.number_input("Inizio BESS", 0, 40, 2),
    'nuc': col_t1.number_input("Inizio Nuc", 0, 40, 15)
}
rate = {
    'pv': col_t2.number_input("Rate PV (GW/anno)", 0.5, 20.0, 6.0, step=0.5),
    'wind': col_t2.number_input("Rate Eol (GW/anno)", 0.1, 10.0, 2.0, step=0.1),
    'bess': col_t2.number_input("Rate BESS (GWh/a)", 0.5, 50.0, 5.0, step=1.0),
    'nuc': col_t2.number_input("Rate Nuc (GW/anno)", 0.2, 3.0, 1.0, step=0.1)
}

st.sidebar.header("⚙️ Mercato & LCA")
mercato = {
    'cfd_pv': st.sidebar.slider("CfD PV (€/MWh)", 20.0, 150.0, 60.0, step=5.0),
    'cfd_wind': st.sidebar.slider("CfD Wind (€/MWh)", 30.0, 150.0, 80.0, step=5.0),
    'cfd_nuc': st.sidebar.slider("CfD Nuc (€/MWh)", 50.0, 200.0, 120.0, step=5.0),
    'bess_capex': st.sidebar.slider("CAPEX BESS (€/MWh)", 50000.0, 300000.0, 120000.0, step=10000.0),
    'wacc_bess': 0.05, 'bess_opex_fix': 0.015, 'bess_vita': 15,
    'gas_eur_mwh': st.sidebar.slider("Gas (€/MWh)", 30.0, 300.0, 130.0, step=10.0),
    'costo_base_integrazione': 10.0, 'voll': 3000.0
}

try:
    cartella_script = os.path.dirname(os.path.abspath(__file__))
    df_completo = carica_dati(
        os.path.join(cartella_script, "dataset_fotovoltaico_produzione.csv"),
        os.path.join(cartella_script, "gme.xlsx"),
        os.path.join(cartella_script, "dataset_eolico_produzione.csv"),
        DEFAULT_PV_NORD_SHARE, DEFAULT_WIND_NORD_SHARE
    )
    
    array_pv = df_completo['Fattore_Capacita_PV'].to_numpy(dtype=np.float64)
    array_wind = df_completo['Fattore_Capacita_Wind'].to_numpy(dtype=np.float64)
    array_fabbisogno = df_completo['Fabbisogno_MW'].to_numpy(dtype=np.float64)
    
    with st.spinner("Calcolo di tutti gli scenari..."):
        risultati_30y = simula_motore_30_anni(array_pv, array_wind, array_fabbisogno, t_start, rate, anni_transizione)

    fabbisogno_annuo_mwh = df_completo['Fabbisogno_MW'].sum()
    miglior_config, df_plot = applica_economia_cumulata(risultati_30y, fabbisogno_annuo_mwh, mercato, anni_transizione)

    # --- RISULTATI ---
    st.subheader(f"🏆 Il Miglior Risultato Raggiunto (all'anno {anni_transizione})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Costo Medio Cumulato", f"{miglior_config['Costo_Medio_30y']:.1f} €/MWh")
    col2.metric("Carbon Intensity Media", f"{miglior_config['Carbon_Intensity_30y']:.1f} gCO₂/kWh")
    
    col3.metric(
        "Risparmio vs Caso 0", 
        f"+{miglior_config['Risparmio_Mld_30y']:.1f} Mld €", 
        help="Miliardi di Euro risparmiati sull'intero sistema rispetto a non costruire nulla (Status Quo)."
    )
    
    st.markdown(
        f"**Capacità effettivamente installata:** {miglior_config['Reached_PV']} GW Solare | "
        f"{miglior_config['Reached_Wind']} GW Eolico | {miglior_config['Reached_BESS']} GWh Batterie | "
        f"{miglior_config['Reached_Nuc']} GW Nucleare\n\n"
        f"*(Spesa Totale del sistema nel periodo: {miglior_config['Spesa_Totale_Mld_30y']:.1f} Mld € | Target che guidava questo scenario: FV {miglior_config['Target_PV']}, Eol {miglior_config['Target_Wind']}, BESS {miglior_config['Target_BESS']}, Nuc {miglior_config['Target_Nuc']})*"
    )

    st.subheader("📊 Frontiera di Pareto")
    fig = px.scatter(
        df_plot, x='Carbon_Intensity_30y', y='Costo_Medio_30y', color='Reached_Nuc',
        color_continuous_scale='Plasma', 
        hover_data=['Reached_PV', 'Reached_Wind', 'Reached_BESS', 'Target_Nuc'],
        labels={
            'Carbon_Intensity_30y': f"Carbon Intensity Media su {anni_transizione} Anni (gCO₂/kWh)",
            'Costo_Medio_30y': f"Costo Medio Bolletta su {anni_transizione} Anni (€/MWh)",
            'Reached_Nuc': "Nucleare Raggiunto (GW)"
        }
    )
    
    fig.add_trace(go.Scatter(
        x=[miglior_config['Carbon_Intensity_30y']], y=[miglior_config['Costo_Medio_30y']],
        mode='markers', marker=dict(color='lime', size=15, line=dict(color='black', width=2)),
        name='Ottimo Scelto'
    ))
    
    fig.update_traces(marker=dict(opacity=0.6, line=dict(width=0.5, color='white')))
    fig.update_layout(xaxis_autorange="reversed", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- TRAIETTORIA DELL'OTTIMO ---
    st.markdown("---")
    st.subheader("🛤️ Traiettoria dell'Ottimo e Consumo di Gas")
    
    storia_t = []
    pv_sq, wind_sq, nuc_sq, bess_sq = 40.0, 10.0, 0.0, 10.0

    for anno in range(anni_transizione + 1):
        pv_gw = get_reached_capacity(anno, t_start['pv'], pv_sq, miglior_config['Target_PV'], rate['pv'])
        wind_gw = get_reached_capacity(anno, t_start['wind'], wind_sq, miglior_config['Target_Wind'], rate['wind'])
        nuc_gw = get_reached_capacity(anno, t_start['nuc'], nuc_sq, miglior_config['Target_Nuc'], rate['nuc'], True)
        bess_gwh = get_reached_capacity(anno, t_start['bess'], bess_sq, miglior_config['Target_BESS'], rate['bess'])
        
        # --- APPLICAZIONE C-RATE ANCHE NELLA TRAIETTORIA UI ---
        bess_mw = bess_gwh * 1000.0 * 0.5
        
        gas, _, _, _, _ = simula_rete_light_fast(
            array_pv, array_wind, array_fabbisogno, pv_gw*1000, wind_gw*1000, nuc_gw*1000, bess_gwh*1000,
            bess_mw, 50000, 2500, 12000, 5000000, 2850
        )
        storia_t.append({'Anno': anno, 'PV_GW': pv_gw, 'Wind_GW': wind_gw, 'Nuc_GW': nuc_gw, 'Gas_TWh': gas/1e6})

    df_t = pd.DataFrame(storia_t)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['PV_GW'], mode='lines', stackgroup='one', name='PV Raggiunto (GW)', fillcolor='gold'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Wind_GW'], mode='lines', stackgroup='one', name='Wind Raggiunto (GW)', fillcolor='lightskyblue'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Nuc_GW'], mode='lines', stackgroup='one', name='Nuc Raggiunto (GW)', fillcolor='mediumpurple'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_t['Anno'], y=df_t['Gas_TWh'], mode='lines+markers', name='Gas (TWh/anno)', line=dict(color='red', width=3)), secondary_y=True)
    
    fig2.update_layout(hovermode="x unified", height=450)
    fig2.update_yaxes(title_text="Capacità Installata (GW)", secondary_y=False)
    max_gas = max(1.0, df_t['Gas_TWh'].max() * 1.2)
    fig2.update_yaxes(title_text="Gas Bruciato (TWh)", secondary_y=True, range=[0, max_gas])
    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # GRAFICO ORARIO DELL'ULTIMO ANNO (ZOOMABILE)
    # ==========================================
    st.markdown("---")
    st.subheader(f"🔍 Dispacciamento Orario (Anno {anni_transizione})")
    st.markdown("Esplora il mix energetico dell'ultimo anno. **Evidenzia col mouse un'area sul grafico per fare zoom** su specifiche settimane o giorni. Usa il doppio clic per tornare alla vista completa.")

    with st.spinner("Calcolo del dispacciamento orario dettagliato..."):
        # Recupero i GW/GWh installati nell'ultimo anno
        pv_gw_fin = get_reached_capacity(anni_transizione, t_start['pv'], pv_sq, miglior_config['Target_PV'], rate['pv'])
        wind_gw_fin = get_reached_capacity(anni_transizione, t_start['wind'], wind_sq, miglior_config['Target_Wind'], rate['wind'])
        nuc_gw_fin = get_reached_capacity(anni_transizione, t_start['nuc'], nuc_sq, miglior_config['Target_Nuc'], rate['nuc'], True)
        bess_gwh_fin = get_reached_capacity(anni_transizione, t_start['bess'], bess_sq, miglior_config['Target_BESS'], rate['bess'])
        bess_mw_fin = bess_gwh_fin * 1000.0 * 0.5

        # Simulo le 8760 ore con la nuova funzione
        (h_pv, h_wind, h_nuc, h_hyd_fl, h_hyd_bac, h_bess, h_gas, h_def) = simula_rete_dettaglio_orario(
            array_pv, array_wind, array_fabbisogno, 
            pv_gw_fin*1000, wind_gw_fin*1000, nuc_gw_fin*1000, bess_gwh_fin*1000, bess_mw_fin,
            50000.0, 2500.0, 12000.0, 5000000.0, 2850.0
        )

        # Costruisco il DataFrame per il plot
        df_orario = pd.DataFrame({
            'Nucleare': h_nuc,
            'Idroelettrico Fluente': h_hyd_fl,
            'Eolico': h_wind,
            'Fotovoltaico': h_pv,
            'Idroelettrico Bacino': h_hyd_bac,
            'Batterie (Scarica)': h_bess,
            'Gas': h_gas,
            'Deficit (Blackout)': h_def,
            'Fabbisogno': array_fabbisogno
        }, index=df_completo.index) # Uso l'indice temporale originale del GME

        # Creo la figura Plotly
        fig3 = go.Figure()

        # Colori personalizzati per una lettura intuitiva
        colori = {
            'Nucleare': '#9b59b6', 'Idroelettrico Fluente': '#3498db',
            'Eolico': '#2ecc71', 'Fotovoltaico': '#f1c40f',
            'Idroelettrico Bacino': '#2980b9', 'Batterie (Scarica)': '#e67e22',
            'Gas': '#e74c3c', 'Deficit (Blackout)': '#000000'
        }

        # Aggiungo le aree impilate (stackgroup)
        for col in colori.keys():
            fig3.add_trace(go.Scatter(
                x=df_orario.index, y=df_orario[col],
                mode='lines', stackgroup='one', name=col, line=dict(width=0.5, color=colori[col])
            ))

        # Aggiungo la linea del Fabbisogno (non impilata, in sovraimpressione)
        fig3.add_trace(go.Scatter(
            x=df_orario.index, y=df_orario['Fabbisogno'],
            mode='lines', name='Fabbisogno', line=dict(color='black', width=2, dash='dot')
        ))

        # Configuro il layout con Range Slider per navigare comodamente
        fig3.update_layout(
            height=600,
            hovermode="x unified",
            yaxis_title="Potenza (MW)",
            legend_title="Fonti",
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.05), # Aggiunge la barra di scorrimento sotto!
                type="date"
            )
        )
        st.plotly_chart(fig3, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Errore: {e}")
