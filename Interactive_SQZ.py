"""
LIGO Squeezing Analysis GUI: Interactive Squeeze!
Quantum noise analysis and visualization
"""

# ============================================================================
# IMPORTS
# ============================================================================
import sys
print(sys.executable)

# ── Standard Libraries ─────────────────────────────────────────────────────────
import os, sys, warnings
import subprocess
import yaml, pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from copy import deepcopy
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

# ── GUI Libraries ─────────────────────────────────────────────────────-

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ── GW-specific packages ───────────────────────────────────────────────────────
import gwpy.timeseries
from gwpy.time import tconvert

# pygwinc lives one level above this gui — add it to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), 'pygwinc'))
cur_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
sys.path.insert(0, cur_dir)

import gwinc, inspiral_range
from gwinc.noise.quantum import shotrad_debug

# ── Measurement parameters ─────────────────────────────────────────────────────-
L = 3995                 # Arm cavity length [m]

# ============================================================================
# AUTH AND DATA PROCESSING FUNCTIONS
# ============================================================================

def check_kerberos_ticket():
    """Check if a valid Kerberos ticket exists"""
    try:
        kinit_path = get_conda_kinit()
        klist_path = kinit_path.replace('kinit', 'klist')
        result = subprocess.run([klist_path, '-s'], capture_output=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking Kerberos ticket: {e}")
        return False

def get_kerberos_status():
    """Get detailed Kerberos ticket status"""
    try:
        kinit_path = get_conda_kinit()
        klist_path = kinit_path.replace('kinit', 'klist')
        result = subprocess.run([klist_path], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, "No Kerberos ticket found"
    except Exception as e:
        return False, f"Error: {str(e)}"
    
def get_conda_kinit():
    """Get path to kinit in conda environment, fallback to system kinit"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        kinit_path = os.path.join(conda_prefix, 'bin', 'kinit')
        if os.path.exists(kinit_path):
            return kinit_path
    # Fallback to system kinit
    return 'kinit'

def authenticate_kerberos(username, password):
    """
    Authenticate with Kerberos using provided credentials
    Returns: (success: bool, message: str)
    """
    try:
        kinit_path = get_conda_kinit()
        # Use kinit with password from stdin
        process = subprocess.Popen(
            [kinit_path, f'{username}@LIGO.ORG'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=password + '\n', timeout=10)
        if process.returncode == 0:
            return True, "Successfully authenticated with Kerberos"
        else:
            error_msg = stderr.strip() if stderr else "Authentication failed"
            return False, error_msg     
    except subprocess.TimeoutExpired:
        process.kill()
        return False, "Authentication timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"

def setup_data_directories():
    """Create gwpy_data directories relative to parent of GUI script"""
    cur_dir  = os.path.abspath(os.path.join(os.getcwd(), "../"))
    meas_dir = os.path.join(cur_dir, 'gwpy_data')
    data_dir = os.path.join(cur_dir, 'gwpy_data', 'data')
    os.makedirs(meas_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return meas_dir, data_dir

def load_pickle(filename):
    """Load pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, filename):
    """Save data to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def get_gps_timestrings(meas_dict,
                        start_str_fmt='%m/%d, %H:%M:%S',
                        stop_str_fmt='%H:%M:%S UTC',
                        start_str_key='gps start',
                        stop_str_key='gps stop'):
    """Convert GPS start/stop times to human-readable strings"""

    start_str = tconvert(meas_dict[start_str_key]).strftime(start_str_fmt)
    stop_str  = tconvert(meas_dict[stop_str_key]).strftime(stop_str_fmt)
    return start_str, stop_str

def load_yaml(path):
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        print(f'Loading YAML from: {path}')
        return yaml.load(f.read(), Loader=yaml.SafeLoader)


# ============================================================================
# DATA ANALYSIS HELPERS
# ============================================================================

# ── Rebin PSDs to a uniform frequency grid ─────────────────────────────────────
def rebin(freq, array, xlen=1000, err=None, quiet=True):
    """Resample (freq, array) onto a log-spaced grid using median binning.

    Parameters
    ----------
    freq  : array — input frequency axis [Hz]
    array : array — data to resample (same length as freq)
    xlen  : int   — number of output bins (default 500)
    err   : array, optional — uncertainty array; if provided, also resampled
                              and reduced by √N per bin
    quiet : bool  — suppress progress print if True

    Returns
    -------
    freq_new, array_new          — if err is None
    freq_new, array_new, err_new — if err is provided
    """
    freq_new, array_new, err_new = [], [], []

    # Build log-spaced target grid and find bin edges in the original array
    freq_target = np.geomspace(min(freq), max(freq), xlen)
    inds = np.searchsorted(freq, freq_target)

    for indL, indR in zip(inds[:-1], inds[1:]):
        bin_width = indR - indL
        if bin_width == 0:
            continue
        freq_new.append(np.nanmean(freq[indL:indR]))
        array_new.append(np.nanmedian(array[indL:indR]))
        if err is not None:
            err_new.append(np.nanmedian(err[indL:indR]) / np.sqrt(bin_width))

    if not quiet: print(f'Rebin → {len(freq_new)} bins')

    if err is not None:
        return np.array(freq_new), np.array(array_new), np.array(err_new)
    return np.array(freq_new), np.array(array_new)

def dB( variance, power=True ): 
    '''  noise variance to dB  '''
    if power: return 10*np.log10(variance)
    else:     return 20*np.log10(variance)

def NLG_to_x( NLG ):
    ''' given NLG, returns pump parameter: x = pump/pump_threshold '''
    return 1 - 1/np.sqrt(NLG)

def NLG_to_genSQZ_dB(NLG):
    ''' given NLG, returns generated sqz level in dB '''
    return dB(R(NLG, 90, mu=[1,0], detuning_param=0))

def R(NLG, sqzang_deg=90, mu=[1,0], detuning_param=0):
    '''returns a qn variance w.r.t. vacuum noise
    by default, calling R( NLG={meas} ) 
    returns generated squeeze dBs corresponding to NLG (no loss, pn, antisqz)
    ---------
    NLG : nonlinear gain
    sqzang_deg : sqz angle in degrees (0=sqz, 90=anti-sqz)
    mu : [eta, pn_rad]
    '''
    eta, pn_rad = mu
    x = NLG_to_x(NLG)
    Rp = 1 + 4*x*eta / ((1-x)**2 + detuning_param**2)
    Rm = 1 - 4*x*eta / ((1+x)**2 + detuning_param**2)
    sqzang_rad  = sqzang_deg/180*np.pi
    qn_variance = Rp*np.sin(sqzang_rad)**2 + Rm*np.cos(sqzang_rad)**2 \
                 + (Rp-Rm)*np.cos(2*sqzang_rad)*np.sin(pn_rad)**2
    return qn_variance

def chop_arrays_to_freq_range(freq, array_to_chop=None,
                              fmin=None, fmax=None,):
    """Trim a frequency array "array_to_chop" to a frequency range (fmin, fmax)"""

    if fmin is None: fmin=min(freq)
    if fmax is None: fmax=max(freq)

    ind_min = np.argmin(np.abs(freq-fmin))
    ind_max = np.argmin(np.abs(freq-fmax))
    chopped_freqs = freq[ind_min:ind_max]
    
    if array_to_chop is not None:
        chopped_array = array_to_chop[ind_min:ind_max]
        return chopped_freqs, chopped_array
    else:
        return chopped_freqs, ind_min, ind_max,

def estimate_median(freq, array, fcenter=925, fband=100):
    """Estimate median of an array for some bandwidth (fband) around center freq (fcenter)"""
    if fcenter is None:
        return np.nanmedian(array)
    else:
        start_idx  = np.argmin( np.abs(freq-(fcenter-fband/2)) )
        stop_idx = np.argmin( np.abs(freq-(fcenter+fband/2)) )
    return np.nanmedian(array[start_idx:stop_idx]), start_idx, stop_idx


def subtract_psds(psd1, psd2, psd1_err, psd2_err):
    subtracted_psd     = psd1 - psd2
    subtracted_psd_err = np.sqrt(psd1_err**2 + psd2_err**2)
    subtracted_asd     = np.nan_to_num(subtracted_psd**0.5)
    subtracted_asd_err = np.nan_to_num(np.sqrt(subtracted_psd + subtracted_psd_err) - np.sqrt(subtracted_psd))
    return subtracted_psd, subtracted_psd_err, subtracted_asd, subtracted_asd_err


def min_sqz_angle(budget, sqz_type, fmin=300, fmax=3500):

    ff = np.linspace(fmin, fmax, 50)
    _budget = gwinc.load_budget('Aplus', bname='Quantum')
    _budget.ifo = deepcopy(budget.ifo)
    _budget.ifo.Squeezer.Type = sqz_type 

    def cost(xx):
        _budget.ifo.Squeezer.SQZAngle = xx[0] * np.pi/180
        return np.sum(_budget.run(freq=ff).asd)
    res = minimize(cost, x0=0, method='Nelder-Mead')
    print(f'Optimal squeezing angle: {res.x[0]:.3f} deg')

    return res.x[0] * np.pi/180


def fit_sqz_dB(budget, meas_ff, meas_dB, sqz_type, guess=0, fmin=500, fmax=3000):
    """Fit squeezing angle [rad] to measured sqz_dB curve using QuantumRelGamma model.

    Parameters
    ----------
    budget  : gwinc budget object
    meas_ff : array — frequency axis [Hz]
    meas_dB : array — measured squeezing in dB
    guess   : float — initial angle guess [deg]
    fmin, fmax : float — frequency band for fitting [Hz]

    Returns
    -------
    sqz_angle : float [rad]
    """
    chopped_ffs, chopped_dBs = chop_arrays_to_freq_range(
        meas_ff, array_to_chop=meas_dB, fmin=fmin, fmax=fmax
    )
    _budget = gwinc.load_budget('Aplus', bname='QuantumRelGamma')
    _budget.ifo = deepcopy(budget.ifo)
    _budget.ifo.Squeezer.Type = sqz_type

    def calc_gwinc(xx):
        _budget.ifo.Squeezer.SQZAngle = np.deg2rad(xx[0])
        return 10 * np.log10(_budget.run(freq=chopped_ffs).psd)

    def cost(xx):
        print('.', end='')
        return np.sum((chopped_dBs - calc_gwinc(xx))**2)

    res = minimize(cost, x0=guess, method='Nelder-Mead')
    print(f'\nFitted sqz angle: {res.x[0]:.3f} deg')
    return np.deg2rad(res.x[0])


def auto_guess_deg(sqz_config, sqz_min_deg):
    """Generate an initial squeezing angle guess from the config key.

    Anchors all guesses to sqz_min_deg (the optimal angle for this dataset)
    so that guesses automatically adapt across different measurements.

    For numeric keys (commanded angles in deg): offset from optimal using the
    empirical relationship between commanded and physical angle.
    For named string keys: offset from optimal based on squeezing type.
    """
    config_str = str(sqz_config).upper()

    #if isinstance(sqz_config, (int, float)):
        #for a numerical key, use the key itself
    #    return float(sqz_config)

    if isinstance(sqz_config, (int, float)):
        #for a numerical key, use the min
        return sqz_min_deg
    
    elif 'AS' in config_str:
        # Anti-squeezing — ~90 deg away from optimal
        return sqz_min_deg + 90
    elif 'MID' in config_str:
        # Mid-squeezing — ~45 deg away from optimal
        return sqz_min_deg + 45
    elif 'FD' in config_str or 'FI' in config_str:
        # Freq-dependent/independent — near optimal
        return sqz_min_deg
    else:
        return sqz_min_deg   # safe fallback
    
# ============================================================================
# GWINC CALL FOR BUDGET
# ============================================================================

# ── Build gwinc noise model for comparison ─────────────────────────────────────

def set_ifo_params(
    bname              = 'Displacement',
    Parm               = 347e3,         # arm cavity power [W]
    readout_eff        = 0.92,          # total readout efficiency (OMC + homodyne)
    sec_detuning_deg   = 0,             # signal recycling cavity detuning [deg]
    homodyne_angle_deg = +10,           # homodyne readout angle [deg]
    sqz_config         = 'None',        # squeezer type: 'None', 'Freq Dependent', etc.
    fcdet_Hz           = -30,           # filter cavity detuning [Hz]
    fc_Tin             = 909e-6,        # filter cavity input mirror transmission
    fc_L               = 297.55,        # filter cavity length [m]
    fc_Lrt             = 50e-6,         # filter cavity round-trip loss
    fc_Lrms            = 1e-12,         # filter cavity length noise RMS [m]
    nlg                = 17,            # nonlinear gain
    sqz_injection_loss = 0.073,         # squeezer path injection loss
    sqz_angle_rms      = 25e-3,         # RMS squeezing angle jitter [rad]
    include_mm         = False,         # include mode-mismatch terms
    MM_IFO_OMC         = 0.0,           # IFO→OMC mismatch amplitude
    MM_IFO_OMCphi      = 0.0,           # IFO→OMC mismatch phase [rad]
    MM_SQZ_OMC         = 0.0,           # squeezer→OMC mismatch amplitude
    MM_SQZ_OMCphi      = 0.0,           # squeezer→OMC mismatch phase [rad]
    FC_L_mm            = 0.0,           # filter cavity length mismatch [m]
    FC_psi_mm          = 0.0,           # filter cavity Gouy phase mismatch [rad]
    ifo_yaml_fname     = 'QuantumParameters_July2025.yaml', # param file 
    quiet              = True,
    use_default_gwinc_budgets = False,
    ):

    # ── Resolve squeezing level ────────────────────────────────────────────────
    gensqz_dB = NLG_to_genSQZ_dB(nlg)

    """Build and configure a gwinc noise budget for H1 O4 parameters.

    Loads the IFO model from H1_O4.yaml if available, otherwise falls back to
    the gwinc Aplus budget with aLIGO optics/materials substituted in.

    Returns
    -------
    budget : gwinc Budget object — call budget.run(freq=freq) to get noise traces
    """

    # ── Load base budget ───────────────────────────────────────────────────────
    if not use_default_gwinc_budgets and os.path.isfile(ifo_yaml_fname):
        # Preferred: use the site-specific H1 yaml for accurate O4 parameters
        budget = gwinc.load_budget(ifo_yaml_fname, bname=bname)
    else:
        # Fallback: Aplus template with aLIGO optics/materials swapped in
        budget = gwinc.load_budget('Aplus', bname=bname)
        aLIGO  = gwinc.load_budget('aLIGO', bname=bname)
        budget.ifo.Optics    = aLIGO.ifo.Optics
        budget.ifo.Materials = aLIGO.ifo.Materials
        # BSLoss is the gwinc code-name for signal extraction cavity (SEC) loss
        budget.ifo.Optics.BSLoss = 3000e-6
        del budget.ifo.Laser.Power
        # SRM06 transmittance from galaxy.ligo.caltech.edu/optics
        budget.ifo.Optics.SRM.Transmittance = 0.3234

    ifo = budget.ifo

    # ── Core IFO parameters ────────────────────────────────────────────────────
    ifo.Laser.ArmPower = Parm
    # Tunephase is the round-trip Gouy phase in gwinc (not the one-way detuning)
    ifo.Optics.SRM.Tunephase                 = np.pi/180 * sec_detuning_deg
    # Quadrature.dc is measured from the amplitude quadrature, so +90° offset
    ifo.Optics.Quadrature.dc                 = np.pi/180 * (homodyne_angle_deg + 90)
    ifo.Optics.PhotoDetectorEfficiency       = readout_eff

    # ── Squeezer parameters ────────────────────────────────────────────────────
    ifo.Squeezer.Type                        = sqz_config
    ifo.Squeezer.AmplitudedB                 = gensqz_dB
    ifo.Squeezer.InjectionLoss               = sqz_injection_loss
    ifo.Squeezer.SQZAngleRMS                 = sqz_angle_rms  # RMS squeezing angle noise [rad]
    
    # ── Filter Cavity ──────────────────────────────────────────────────────────
    ifo.Squeezer.FilterCavity.fdetune        = fcdet_Hz       # detuning 
    ifo.Squeezer.FilterCavity.L              = fc_L           # filter cavity length [m]
    ifo.Squeezer.FilterCavity.Ti             = fc_Tin         # input mirror transmission
    ifo.Squeezer.FilterCavity.Lrt            = fc_Lrt         # round-trip loss
    ifo.Squeezer.FilterCavity.Lrms           = fc_Lrms        # RMS length noise [m]

    # ── Mode-mismatch terms ────────────────────────────────────────────────────
    # Always set unconditionally — zero when include_mm=False so no stale values
    # direct_mm_sqz_ifo is always False — controls internal gwinc MM routing.
    ifo.Squeezer.direct_mm_sqz_ifo  = False
    ifo.Optics.MM_IFO_OMC           = MM_IFO_OMC    if include_mm else 0.0
    ifo.Optics.MM_IFO_OMCphi        = MM_IFO_OMCphi if include_mm else 0.0
    ifo.Squeezer.MM_SQZ_OMC         = MM_SQZ_OMC    if include_mm else 0.0
    ifo.Squeezer.MM_SQZ_OMCphi      = MM_SQZ_OMCphi if include_mm else 0.0
    ifo.Squeezer.FilterCavity.L_mm  = FC_L_mm       if include_mm else 0.0
    ifo.Squeezer.FilterCavity.psi_mm = FC_psi_mm    if include_mm else 0.0

    # ── Diagnostic printout (quiet=False) ──────────────────────────────────────
    if not quiet:
        print(f'{ifo.Laser.ArmPower/1e3 = } kW')
        print(f'{ifo.Optics.PhotoDetectorEfficiency = }')
        print(f'{ifo.Optics.SRM.Tunephase = }')
        print(f'{(ifo.Optics.Quadrature.dc - np.pi/2)*180/np.pi = :0.3}')
        print(f'{ifo.Squeezer.Type = }')
        print(f'{ifo.Squeezer.InjectionLoss = }')
        print(f'{ifo.Squeezer.SQZAngleRMS = }')
        print(f'{ifo.Squeezer.FilterCavity.L = }')
        print(f'{ifo.Squeezer.FilterCavity.Lrt = }')
        print(f'{ifo.Squeezer.FilterCavity.Lrms = }')
        if include_mm:
            print(f'{ifo.Optics.MM_IFO_OMC = }  → {ifo.Optics.MM_IFO_OMC**2/2*100:.3f}% power loss')
            print(f'{ifo.Optics.MM_IFO_OMCphi = } rad')
            print(f'{ifo.Squeezer.MM_SQZ_OMC = }  → {ifo.Squeezer.MM_SQZ_OMC**2/2*100:.3f}% power loss')
            print(f'{ifo.Squeezer.MM_SQZ_OMCphi = } rad')
            print(f'{ifo.Squeezer.FilterCavity.L_mm = }')
            print(f'{ifo.Squeezer.FilterCavity.psi_mm = }')

    return budget

# ============================================================================
# INITIALIZE APP & STYLES
# ============================================================================

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Color scheme 
COLORS = {
    'background': '#FFFFFF',
    'surface': '#F8F9FA',
    'primary': '#2C3E50',
    'secondary': '#34495E',
    'accent': '#3498DB',
    'text': '#2C3E50',
    'text_secondary': '#7F8C8D',
    'border': '#DEE2E6',
    'success': '#27AE60',
    'warning': '#F39C12',
    'error': '#E74C3C'
}

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '320px',
    'padding': '20px',
    'backgroundColor': COLORS['surface'],
    'borderRight': f"1px solid {COLORS['border']}",
    'overflowY': 'auto'
}

CONTENT_STYLE = {
    'marginLeft': '340px',
    'marginRight': '20px',
    'padding': '20px',
}

CARD_STYLE = {
    'marginBottom': '15px',
    'boxShadow': '0 1px 3px rgba(0,0,0,0.12)',
    'border': f"1px solid {COLORS['border']}"
}

BUTTON_STYLE = {
    'width': '100%',
    'marginTop': '10px',
    'fontWeight': '500'
}

# ============================================================================
# COMPONENTS
# ============================================================================

def create_header():
    """Create application header"""
    return html.Div([
        html.H3('LIGO Squeezing Analysis', 
                style={'marginBottom': '5px', 'color': COLORS['primary']}),
        html.P('Quantum Noise Characterization and Modeling',
               style={'color': COLORS['text_secondary'], 'fontSize': '14px', 'marginBottom': '0'})
    ], style={'marginBottom': '30px'})


def create_data_fetch_section():
    """Data fetching and configuration controls"""
    return dbc.Card([
        dbc.CardHeader(html.H5('Data Configuration', className='mb-0')),
        dbc.CardBody([
            html.Label('Configuration File', style={'fontWeight': '500', 'fontSize': '13px'}),
            dcc.Dropdown(
                id='config-file-dropdown',
                options=[{'label': 'gps_reference_times.yml', 'value': 'gps_reference_times.yml'}],
                value='gps_reference_times.yml',
                clearable=False,
                style={'marginBottom': '15px'}
            ),
            html.Label('Dataset', style={'fontWeight': '500', 'fontSize': '13px'}),
            dcc.Dropdown(
                id='dataset-dropdown',
                placeholder='Select dataset...',
                style={'marginBottom': '15px'}
            ),
            # ← dcc.Store(id='dataset-config-store') REMOVED from here
            dbc.Checkbox(
                id='reprocess-checkbox',
                label='Force reprocess (re-fetch from NDS)',
                value=False,
                style={'marginBottom': '10px', 'fontSize': '13px'}
            ),
            dbc.Button('Load Data', id='fetch-data-button', color='primary', style=BUTTON_STYLE),
            html.Div(id='fetch-status', style={'marginTop': '15px'}),
            dbc.Progress(id='fetch-progress', value=0, style={'marginTop': '10px'})
        ])
    ], style=CARD_STYLE)


def create_analysis_parameters_section():
    return dbc.Card([
        dbc.CardHeader(html.H5('Analysis Parameters', className='mb-0')),
        dbc.CardBody([
            html.P(
                'Values loaded from YAML — edit to override before running analysis.',
                style={'fontSize': '12px', 'color': COLORS['text_secondary'], 'marginBottom': '12px'}
            ),
            dbc.Accordion([
                # ── Interferometer ────────────────────────────────────────────
                dbc.AccordionItem([
                    html.Label('IFO Parameter File', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='ifo-yaml-fname-input', type='text',
                              value='QuantumParameters_July2025.yaml',
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('Arm Power (W)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='arm-power-input', type='number', value=347000,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('Homodyne Angle (deg)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='homodyne-angle-input', type='number', value=10.0, step=0.1,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('Readout Efficiency', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='readout-eff-input', type='number', value=0.92, step=0.01,
                              min=0, max=1, style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('SEC Detuning (deg)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='sec-detuning-input', type='number', value=0, step=0.1,
                              style={'width': '100%', 'marginBottom': '15px'}),
                ], title='Interferometer', item_id='interferometer'),

                # ── Squeezer ──────────────────────────────────────────────────
                dbc.AccordionItem([
                    html.Label('NLG (Nonlinear Gain)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='nlg-input', type='number', value=24.0,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('Squeezer Type', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='sqz-type-dropdown',
                        options=[
                            {'label': 'Freq Dependent',   'value': 'Freq Dependent'},
                            {'label': 'Freq Independent', 'value': 'Freq Independent'},
                            {'label': 'None',             'value': 'None'},
                        ],
                        value='Freq Dependent',
                        clearable=False,
                        style={'marginBottom': '15px'}
                    ),
                    html.Label('Injection Loss', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='sqz-injection-loss-input', type='number',
                              value=0.073, min=0, max=0.5, step=0.001,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('SQZ Angle RMS (rad)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='sqz-angle-rms-input', type='number',
                              value=0.025, min=0, max=0.1, step=0.001,
                              style={'width': '100%', 'marginBottom': '15px'}),
                ], title='Squeezer', item_id='squeezer'),

                # ── Filter Cavity ─────────────────────────────────────────────
                dbc.AccordionItem([
                    html.Label('FC Detuning (Hz)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='fc-detuning-input', type='number', value=-30,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('FC Input Mirror Transmission', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='fc-tin-input', type='number', value=909e-6, step=1e-6,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('FC Length (m)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='fc-length-input', type='number', value=297.55,
                              min=100, max=400, step=0.01,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('FC Round-Trip Loss', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='fc-lrt-input', type='number', value=50e-6,
                              min=0, max=500e-6, step=1e-6,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('FC Length Noise RMS (m)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='fc-lrms-input', type='number', value=1e-12,
                              min=0, max=10e-12, step=1e-13,
                              style={'width': '100%', 'marginBottom': '15px'}),
                ], title='Filter Cavity', item_id='filter_cavity'),

                # ── Mode Mismatch ─────────────────────────────────────────────
                dbc.AccordionItem([
                    dbc.Checkbox(
                        id='include-mm-checkbox',
                        label='Enable mode-mismatch terms',
                        value=False,
                        style={'marginBottom': '12px', 'fontSize': '13px'}
                    ),
                    html.Label('MM IFO→OMC amplitude', style={'fontWeight': '500', 'fontSize': '13px'}),
                    html.Small('power loss = MM²/2', style={
                        'color': COLORS['text_secondary'], 'display': 'block',
                        'marginBottom': '4px', 'fontSize': '11px'}),
                    dcc.Input(id='mm-ifo-omc-input', type='number', value=0.0,
                              min=0, max=0.5, step=0.01,
                              style={'width': '100%', 'marginBottom': '4px'}),
                    html.Div(id='mm-ifo-omc-loss-display',
                             style={'fontSize': '11px', 'color': COLORS['text_secondary'],
                                    'marginBottom': '12px'}),
                    html.Label('MM IFO→OMC phase (rad)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='mm-ifo-omcphi-input', type='number', value=0.0,
                              min=0, max=np.pi, step=0.05,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('MM squeezer→OMC amplitude', style={'fontWeight': '500', 'fontSize': '13px'}),
                    html.Small('power loss = MM²/2', style={
                        'color': COLORS['text_secondary'], 'display': 'block',
                        'marginBottom': '4px', 'fontSize': '11px'}),
                    dcc.Input(id='mm-sqz-omc-input', type='number', value=0.0,
                              min=0, max=0.5, step=0.01,
                              style={'width': '100%', 'marginBottom': '4px'}),
                    html.Div(id='mm-sqz-omc-loss-display',
                             style={'fontSize': '11px', 'color': COLORS['text_secondary'],
                                    'marginBottom': '12px'}),
                    html.Label('MM squeezer→OMC phase (rad)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='mm-sqz-omcphi-input', type='number', value=0.0,
                              min=0, max=np.pi, step=0.05,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('FC Length Mismatch (m)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='fc-l-mm-input', type='number', value=0.0,
                              min=0, max=1.0, step=0.01,
                              style={'width': '100%', 'marginBottom': '15px'}),
                    html.Label('FC Gouy Phase Mismatch (rad)', style={'fontWeight': '500', 'fontSize': '13px'}),
                    dcc.Input(id='fc-psi-mm-input', type='number', value=0.0,
                              min=0, max=np.pi, step=0.01,
                              style={'width': '100%', 'marginBottom': '15px'}),
                ], title='Mode Mismatch', item_id='mode_mismatch'),

            ], start_collapsed=True, always_open=True),
            html.Div([
                dbc.Button('Run Analysis', id='update-analysis-button',
                           color='success', style=BUTTON_STYLE)
            ], style={'marginTop': '15px'})
        ])
    ], style=CARD_STYLE)


def create_plot_controls_section():
    """Plot display controls"""
    return dbc.Card([
        dbc.CardHeader(html.H5('Display Options', className='mb-0')),
        dbc.CardBody([
            html.Label('Frequency Range (Hz)', style={'fontWeight': '500', 'fontSize': '13px'}),
            dcc.RangeSlider(
                id='freq-range-slider',
                min=1, max=4,
                value=[1.3, 3.8],
                marks={i: f'10^{i}' for i in range(1, 5)},
                step=0.1,
                tooltip={'placement': 'bottom', 'always_visible': False}
            ),
            html.Div(style={'marginBottom': '20px'}),
            html.Label('Rebin Bins', style={'fontWeight': '500', 'fontSize': '13px'}),
            dcc.Input(
                id='rebin-length-input',
                type='number',
                value=1000,
                min=100, max=5000, step=100,
                style={'width': '100%'}
            ),
        ])
    ], style=CARD_STYLE)

# ============================================================================
# MAIN LAYOUT
# ============================================================================

sidebar = html.Div([
    create_header(),
    create_data_fetch_section(),
    create_analysis_parameters_section(),
    create_plot_controls_section()
], style=SIDEBAR_STYLE)


main_content = html.Div([
    dbc.Tabs([
        dbc.Tab(label='Data Overview',           tab_id='tab-overview'),
        dbc.Tab(label='Noise Budget',            tab_id='tab-budget'),
        dbc.Tab(label='Squeezing Analysis',      tab_id='tab-squeezing'),
    ], id='tabs', active_tab='tab-overview', style={'marginBottom': '20px'}),

    dbc.Card([
        dbc.CardBody([
            dcc.Loading(
                id='loading-plots',
                type='circle',
                children=[html.Div(id='plot-container')]
            )
        ])
    ], style={'minHeight': '600px'}),

    dbc.Card([
        dbc.CardBody([
            html.H6('Analysis Information', style={'marginBottom': '10px'}),
            html.Div(id='info-panel', style={'fontSize': '13px', 'fontFamily': 'monospace'})
        ])
    ], style={'marginTop': '20px'})

], style=CONTENT_STYLE)


app.layout = html.Div([
    # Kerberos Authentication Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle('Kerberos Authentication Required')),
        dbc.ModalBody([
            html.P('No cached data found. Please authenticate to fetch data from NDS server.'),
            dbc.Label('LIGO Username (without @LIGO.ORG):'),
            dbc.Input(id='kerberos-username', type='text', placeholder='firstname.lastname'),
            html.Br(),
            dbc.Label('Password:'),
            dbc.Input(id='kerberos-password', type='password', placeholder='Enter password'),
            html.Div(id='kerberos-status', style={'marginTop': '10px', 'color': 'red'})
        ]),
        dbc.ModalFooter([
            dbc.Button('Close', id='kerberos-cancel', color='secondary', className='me-2'),
            dbc.Button('Authenticate', id='kerberos-submit', color='primary')
        ])
    ], id='kerberos-modal', is_open=False),

    # Data stores
    dcc.Store(id='dataset-store'),           # holds meas_dict after load
    dcc.Store(id='analysis-results-store'),  # holds status after analysis runs
    dcc.Store(id='config-data-store'),       # holds full gps_dict_all from YAML
    dcc.Store(id='dataset-config-store'),    # holds single dataset entry from YAML

    sidebar,
    main_content
])


# ============================================================================
# CALLBACKS - DATA LOADING
# ============================================================================

FIXED_CHANS = [
    'H1:SQZ-CLF_REFL_RF6_PHASE_PHASEDEG',
    'H1:AWC-ZM4_PSAMS_VOLTAGE_DC',
    'H1:AWC-ZM4_PSAMS_STRAIN_VOLTAGE',
    'H1:AWC-ZM5_PSAMS_VOLTAGE_DC',
    'H1:AWC-ZM5_PSAMS_STRAIN_VOLTAGE',
    'H1:CDS-SENSMON_CAL_SNSW_EFFECTIVE_RANGE_MPC',
]

SKIP_KEYS = {'span', 'darm_chan', 'params', 'nlg', 'sqz_type', 'budget_kwargs'}


@app.callback(
    [Output('dataset-dropdown', 'options'),
     Output('config-data-store', 'data')],
    Input('config-file-dropdown', 'value')
)
def load_config_file(config_filename):
    """Load YAML config and populate dataset dropdown"""
    try:
        if not config_filename or not os.path.exists(config_filename):
            return [], {}
        config_data = load_yaml(config_filename)
        options = [{'label': name, 'value': name} for name in config_data.keys()]
        return options, config_data
    except Exception as e:
        print(f"Error loading config file: {e}")
        return [], {}


@app.callback(
    [Output('dataset-config-store',      'data'),
     # Interferometer
     Output('arm-power-input',           'value'),
     Output('homodyne-angle-input',      'value'),
     Output('readout-eff-input',         'value'),
     Output('sec-detuning-input',        'value'),
     # Squeezer
     Output('nlg-input',                 'value'),
     Output('sqz-type-dropdown',         'value'),
     Output('sqz-injection-loss-input',  'value'),
     Output('sqz-angle-rms-input',       'value'),
     # Filter Cavity
     Output('fc-detuning-input',         'value'),
     Output('fc-tin-input',              'value'),
     Output('fc-length-input',           'value'),
     Output('fc-lrt-input',              'value'),
     Output('fc-lrms-input',             'value'),
     # IFO yaml
     Output('ifo-yaml-fname-input',      'value'),
     # Mode Mismatch
     Output('include-mm-checkbox',       'value'),
     Output('mm-ifo-omc-input',          'value'),
     Output('mm-ifo-omcphi-input',       'value'),
     Output('mm-sqz-omc-input',          'value'),
     Output('mm-sqz-omcphi-input',       'value'),
     Output('fc-l-mm-input',             'value'),
     Output('fc-psi-mm-input',           'value')],
    Input('dataset-dropdown', 'value'),
    State('config-data-store', 'data')
)
def store_dataset_config(dataset_name, config_data):
    defaults = (
        None,
        347000, 10.0, 0.92, 0,
        24.0, 'Freq Dependent', 0.073, 0.025,
        -30, 909e-6, 297.55, 50e-6, 1e-12,
        'QuantumParameters_July2025.yaml',
        False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    if not dataset_name or not config_data:
        return defaults

    dataset_cfg = config_data.get(dataset_name, {})
    kwargs      = dataset_cfg.get('budget_kwargs', {})

    return (
        dataset_cfg,
        # Interferometer
        kwargs.get('Parm',                347000),
        kwargs.get('homodyne_angle_deg',  10.0),
        kwargs.get('readout_eff',         0.92),
        kwargs.get('sec_detuning_deg',    0),
        # Squeezer
        kwargs.get('nlg',                 24.0),
        kwargs.get('sqz_config',          'Freq Dependent'),
        kwargs.get('sqz_injection_loss',  0.073),
        kwargs.get('sqz_angle_rms',       0.025),
        # Filter Cavity
        kwargs.get('fcdet_Hz',            -30),
        kwargs.get('fc_Tin',              909e-6),
        kwargs.get('fc_L',                297.55),
        kwargs.get('fc_Lrt',              50e-6),
        kwargs.get('fc_Lrms',             1e-12),
        # IFO yaml
        kwargs.get('ifo_yaml_fname',      'QuantumParameters_July2025.yaml'),
        # Mode Mismatch
        kwargs.get('include_mm',          False),
        kwargs.get('MM_IFO_OMC',          0.0),
        kwargs.get('MM_IFO_OMCphi',       0.0),
        kwargs.get('MM_SQZ_OMC',          0.0),
        kwargs.get('MM_SQZ_OMCphi',       0.0),
        kwargs.get('FC_L_mm',             0.0),
        kwargs.get('FC_psi_mm',           0.0),
    )

@app.callback(
    [Output('fetch-status',        'children'),
     Output('fetch-progress',      'value'),
     Output('dataset-store',       'data'),
     Output('kerberos-modal',      'is_open')],
    Input('fetch-data-button', 'n_clicks'),
    [State('dataset-dropdown',     'value'),
     State('dataset-config-store', 'data'),
     State('reprocess-checkbox',   'value')]
)
def fetch_or_load_data(n_clicks, dataset_name, dataset_config, reprocess):
    """
    Load meas_dict for the selected dataset.
    Priority: 1) pickle (unless reprocess=True), 2) local HDF5, 3) NDS fetch
    Kerberos is only checked if we actually need to hit NDS.
    """
    if not n_clicks or not dataset_name or not dataset_config:
        return '', 0, None, False

    try:
        meas_dir, data_dir = setup_data_directories()
        pickle_fname = os.path.join(meas_dir, f'{dataset_name}.pkl')

        # ── 1. Pickle exists and reprocess not forced — load immediately ──────
        # No Kerberos check needed at all in this path
        if not reprocess and os.path.isfile(pickle_fname):
            meas_dict = load_pickle(pickle_fname)
            status = dbc.Alert(
                f"Loaded cached data for {dataset_name}",
                color='success'
            )
            return status, 100, meas_dict, False

        # ── 2. No pickle — check whether any HDF5 files are missing ──────────
        # If every config has a local HDF5 we can process without NDS at all.
        # Only open the Kerberos modal if at least one HDF5 is absent.
        sqz_configs = [k for k in dataset_config if k not in SKIP_KEYS]
        total       = len(sqz_configs)

        missing_hdf5 = []
        for sqz_config in sqz_configs:
            start_time = dataset_config[sqz_config]
            duration   = dataset_config['span']
            fname = os.path.join(
                data_dir,
                f'{sqz_config}_start{start_time}_span{duration}.hdf5'
            ).replace(' ', '_')
            if not os.path.isfile(fname):
                missing_hdf5.append((sqz_config, fname))

        if missing_hdf5 and not check_kerberos_ticket():
            # Open the modal and stop — fetch_or_load_data will re-fire
            # after the user authenticates and clicks Load Data again
            return '', 0, None, True

        # ── 3. Process all configs ────────────────────────────────────────────
        darm_chan_name = dataset_config['darm_chan']
        chans          = [darm_chan_name] + FIXED_CHANS
        meas_dict      = {dataset_name: {}}

        for i, sqz_config in enumerate(sqz_configs):
            print(f'\n  Config: {sqz_config} ({i+1}/{total})')

            start_time = dataset_config[sqz_config]
            duration   = dataset_config['span']
            stop_time  = start_time + duration

            meas_dict[dataset_name][sqz_config] = {
                'gps start': start_time,
                'gps stop':  stop_time,
            }

            print(f'  {tconvert(start_time).strftime("%m/%d/%Y, %H:%M:%S")} '
                  f'– {tconvert(stop_time).strftime("%H:%M:%S UTC")}')

            fname = os.path.join(
                data_dir,
                f'{sqz_config}_start{start_time}_span{duration}.hdf5'
            ).replace(' ', '_')

            # ── Try local HDF5 first ──────────────────────────────────────────
            try:
                print(f'  Looking for local HDF5: {fname}')
                data = gwpy.timeseries.TimeSeriesDict.read(fname, format='hdf5')
                print('  Loaded from disk.')
            except Exception:
                # ── Fetch from NDS ────────────────────────────────────────────
                # We already confirmed a ticket exists above, so this should work
                print('  Fetching from NDS...')
                try:
                    data = gwpy.timeseries.TimeSeriesDict.fetch(
                        chans, start_time, stop_time,
                        host='nds.ligo-wa.caltech.edu', port=31200, verbose=True
                    )
                except Exception as e:
                    status = dbc.Alert([
                        html.Strong("NDS fetch failed"), html.Br(),
                        html.Span(str(e), style={'fontSize': '12px'})
                    ], color='danger')
                    return status, int(i / total * 100), None, False

            data.write(fname, overwrite=True)
            print(f'  Saved HDF5: {fname}')

            # ── Compute PSD / ASD ─────────────────────────────────────────────
            psd_obj     = data[chans[0]].psd(fftlength=5, window='hann', method='median')
            freq        = psd_obj.frequencies.value
            binwidth_Hz = psd_obj.df.value
            psd         = psd_obj.value
            psd_err     = psd / np.sqrt(2 * binwidth_Hz * duration)
            asd         = np.sqrt(psd)
            asd_err     = np.sqrt(psd + psd_err) - np.sqrt(psd)

            _, ind_min, ind_max = chop_arrays_to_freq_range(freq, fmin=1, fmax=7e3)

            meas_dict[dataset_name][sqz_config].update({
                'freq':               freq[ind_min:ind_max],
                'binwidth_Hz':        binwidth_Hz,
                'GDS strain':         asd[ind_min:ind_max],
                'GDS meters':         asd[ind_min:ind_max] * L,
                'GDS meters err':     asd_err[ind_min:ind_max] * L,
                'GDS meters psd':     psd[ind_min:ind_max] * (L**2),
                'GDS meters psd err': psd_err[ind_min:ind_max] * (L**2),
            })

            for chan in chans:
                if 'GDS' not in chan:
                    meas_dict[dataset_name][sqz_config][chan] = np.median(data[chan].value)

        # ── Save pickle ───────────────────────────────────────────────────────
        save_pickle(meas_dict, pickle_fname)

        n_fetched = len(missing_hdf5)
        msg = (f"Fetched {n_fetched} segment(s) from NDS, "
               f"{total - n_fetched} loaded from disk — {dataset_name}"
               if n_fetched else
               f"Processed from local HDF5 files — {dataset_name}")
        return dbc.Alert(msg, color='success'), 100, meas_dict, False

    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Error: {str(e)}", color='danger'), 0, None, False

# ============================================================================
# CALLBACKS - ANALYSIS
# ============================================================================

_cached_traces = {}

@app.callback(
    Output('analysis-results-store', 'data'),
    Input('update-analysis-button',      'n_clicks'),
    [State('dataset-store',              'data'),
     State('dataset-dropdown',           'value'),
     # Interferometer
     State('arm-power-input',            'value'),
     State('homodyne-angle-input',       'value'),
     State('readout-eff-input',          'value'),
     State('sec-detuning-input',         'value'),
     # Squeezer
     State('nlg-input',                  'value'),
     State('sqz-type-dropdown',          'value'),
     State('sqz-injection-loss-input',   'value'),
     State('sqz-angle-rms-input',        'value'),
     # Filter Cavity
     State('fc-detuning-input',          'value'),
     State('fc-tin-input',               'value'),
     State('fc-length-input',            'value'),
     State('fc-lrt-input',               'value'),
     State('fc-lrms-input',              'value'),
     # IFO yaml
     State('ifo-yaml-fname-input',       'value'),
     # Mode Mismatch
     State('include-mm-checkbox',        'value'),
     State('mm-ifo-omc-input',           'value'),
     State('mm-ifo-omcphi-input',        'value'),
     State('mm-sqz-omc-input',           'value'),
     State('mm-sqz-omcphi-input',        'value'),
     State('fc-l-mm-input',              'value'),
     State('fc-psi-mm-input',            'value')]
)
def run_analysis(n_clicks, dataset, dataset_name,
                 arm_power, hd_angle, readout_eff, sec_detuning,
                 nlg, sqz_type, sqz_injection_loss, sqz_angle_rms,
                 fc_detuning, fc_tin, fc_length, fc_lrt, fc_lrms,
                 ifo_yaml_fname,
                 include_mm, mm_ifo_omc, mm_ifo_omcphi,
                 mm_sqz_omc, mm_sqz_omcphi, fc_l_mm, fc_psi_mm):

    if not n_clicks or not dataset:
        return None
    global _cached_traces
    try:
        meas_dict = deepcopy(dataset)
        darm_chan  = 'GDS meters'
        fcenter    = 1300
        inner      = meas_dict[dataset_name]

        for sqz_config in inner:
            if isinstance(inner[sqz_config], dict) and 'freq' in inner[sqz_config]:
                inner[sqz_config]['freq'] = np.array(inner[sqz_config]['freq'])
                for key in ['GDS meters', 'GDS meters err',
                            'GDS meters psd', 'GDS meters psd err']:
                    if key in inner[sqz_config]:
                        inner[sqz_config][key] = np.array(inner[sqz_config][key])

        freq = inner['nosqz']['freq']

        # ── Step 1: build budget from GUI params ──────────────────────────────
        budget = set_ifo_params(
            Parm               = arm_power,
            homodyne_angle_deg = hd_angle,
            readout_eff        = readout_eff,
            sec_detuning_deg   = sec_detuning,
            sqz_config         = sqz_type,
            fcdet_Hz           = fc_detuning,
            fc_Tin             = fc_tin             if fc_tin             is not None else 909e-6,
            fc_L               = fc_length          if fc_length          is not None else 297.55,
            fc_Lrt             = fc_lrt             if fc_lrt             is not None else 50e-6,
            fc_Lrms            = fc_lrms            if fc_lrms            is not None else 1e-12,
            nlg                = nlg,
            sqz_injection_loss = sqz_injection_loss if sqz_injection_loss is not None else 0.073,
            sqz_angle_rms      = sqz_angle_rms      if sqz_angle_rms      is not None else 25e-3,
            ifo_yaml_fname     = ifo_yaml_fname     if ifo_yaml_fname     is not None else 'QuantumParameters_July2025.yaml',
            include_mm         = include_mm         if include_mm         is not None else False,
            MM_IFO_OMC         = mm_ifo_omc         if mm_ifo_omc         is not None else 0.0,
            MM_IFO_OMCphi      = mm_ifo_omcphi      if mm_ifo_omcphi      is not None else 0.0,
            MM_SQZ_OMC         = mm_sqz_omc         if mm_sqz_omc         is not None else 0.0,
            MM_SQZ_OMCphi      = mm_sqz_omcphi      if mm_sqz_omcphi      is not None else 0.0,
            FC_L_mm            = fc_l_mm            if fc_l_mm            is not None else 0.0,
            FC_psi_mm          = fc_psi_mm          if fc_psi_mm          is not None else 0.0,
            quiet              = True,
        )
        ifo = budget.ifo

        # ── Step 2: lossless nosqz model ──────────────────────────────────────
        ifo.Squeezer.Type = 'None'
        ifo.Optics.PhotoDetectorEfficiency = 1
        traces_noloss  = budget.run(freq=freq, ifo=ifo)
        nosqz_calc_asd = traces_noloss.asd

        # ── Step 3: estimate readout efficiency ───────────────────────────────
        nosqz_meas_asd = inner['nosqz'][darm_chan]
        median_readout_eff, _, _ = estimate_median(
            freq,
            (nosqz_calc_asd / nosqz_meas_asd * 1.01) ** 2,
            fcenter=fcenter, fband=100
        )
        median_readout_eff = np.round(median_readout_eff, 2)
        print(f'  Estimated readout efficiency: {median_readout_eff}')

        # ── Step 4: lossy nosqz model ─────────────────────────────────────────
        ifo.Optics.PhotoDetectorEfficiency = median_readout_eff
        traces_nosqz = budget.run(freq=freq, ifo=ifo)
        nosqz_lossy_qn_model_psd = traces_nosqz['Quantum'].psd

        # ── Step 5: technical noise ───────────────────────────────────────────
        nosqz_meas_psd     = inner['nosqz'][darm_chan + ' psd']
        nosqz_meas_psd_err = inner['nosqz'][darm_chan + ' psd err']
        tn_psd, tn_psd_err, tn_asd, tn_asd_err = subtract_psds(
            psd1=nosqz_meas_psd, psd2=nosqz_lossy_qn_model_psd,
            psd1_err=nosqz_meas_psd_err, psd2_err=0
        )

        # ── Step 6: sqz_dB for every config ───────────────────────────────────
        skip     = {'NLG'}
        sqz_keys = [k for k in inner if k not in skip
                    and isinstance(inner[k], dict)
                    and 'freq' in inner[k]]

        for sqz_config in sqz_keys:
            sqz_psd     = inner[sqz_config][darm_chan + ' psd']
            sqz_psd_err = inner[sqz_config][darm_chan + ' psd err']
            sqz_qn_psd, sqz_qn_psd_err, sqz_qn_asd, sqz_qn_asd_err = subtract_psds(
                psd1=sqz_psd, psd2=tn_psd,
                psd1_err=sqz_psd_err, psd2_err=tn_psd_err
            )
            sqz_qnr_dB = np.nan_to_num(
                10 * np.log10(sqz_qn_psd / nosqz_lossy_qn_model_psd)
            )
            inner[sqz_config]['sqz qn asd']     = sqz_qn_asd
            inner[sqz_config]['sqz qn asd err'] = sqz_qn_asd_err
            inner[sqz_config]['sqz_dB']         = sqz_qnr_dB

        inner['nosqz']['ifo output loss'] = median_readout_eff
        inner['nosqz']['tn psd']          = tn_psd
        inner['nosqz']['tn psd err']      = tn_psd_err

        # ── Step 7: QuantumRelGamma budget ────────────────────────────────────
        budget_relgamma     = gwinc.load_budget('Aplus', bname='QuantumRelGamma')
        budget_relgamma.ifo = deepcopy(ifo)

        # ── Step 8: optimal sqz angle ─────────────────────────────────────────
        sqz_min_rad = min_sqz_angle(budget, sqz_type=sqz_type)
        ifo.Squeezer.Type     = sqz_type
        ifo.Squeezer.SQZAngle = sqz_min_rad
        traces_sqz = budget.run(freq=freq, ifo=ifo)

        # ── Step 9: colors ────────────────────────────────────────────────────
        cmap   = cm.tab20
        n      = max(len(sqz_keys) - 1, 1)
        colors = {k: '#{:02x}{:02x}{:02x}'.format(
                      int(cmap(i/n)[0]*255),
                      int(cmap(i/n)[1]*255),
                      int(cmap(i/n)[2]*255))
                  for i, k in enumerate(sqz_keys)}
        colors['nosqz'] = 'black'

        # ── Cache ─────────────────────────────────────────────────────────────
        _cached_traces = {
            'dataset_name':    dataset_name,
            'freq':            freq,
            'nosqz_calc_asd':  nosqz_calc_asd,   # lossless — for budget plot
            'traces_nosqz':    traces_nosqz,
            'traces_sqz':      traces_sqz,
            'meas_dict':       {dataset_name: inner},
            'sqz_keys':        sqz_keys,
            'colors':          colors,
            'budget':          budget,
            'budget_relgamma': budget_relgamma,
            'ifo':             ifo,
            'sqz_type':        sqz_type,
            'sqz_min_rad':     sqz_min_rad,
            'darm_chan':       darm_chan,
        }
        print(f'✓ Analysis complete — {len(sqz_keys)} configs: {sqz_keys}')
        return {'status': 'complete', 'dataset': dataset_name}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}


# ============================================================================
# CALLBACKS - VISUALIZATION
# ============================================================================

@app.callback(
    Output('plot-container', 'children'),
    [Input('tabs',                  'active_tab'),
     Input('analysis-results-store','data'),
     Input('freq-range-slider',     'value'),
     Input('rebin-length-input',    'value')],
    [State('dataset-store', 'data')]
)

def update_plots(active_tab, analysis_results, freq_range, rebin_length, dataset):
    """Route to the correct plot function based on active tab"""

    freq_min = 10 ** freq_range[0]
    freq_max = 10 ** freq_range[1]
    xlen     = rebin_length or 1000

    if active_tab == 'tab-overview':
        if not dataset:
            return html.Div("Load data to view plots",
                style={'textAlign': 'center', 'padding': '50px',
                       'color': COLORS['text_secondary']})
        fig = create_overview_plot(dataset, freq_min, freq_max, xlen)

    else:
        if not analysis_results or not dataset:
            return html.Div("Load data and run analysis to view plots",
                style={'textAlign': 'center', 'padding': '50px',
                       'color': COLORS['text_secondary']})
        if active_tab == 'tab-budget':
            fig = create_noise_budget_plot(freq_min, freq_max, xlen)
        elif active_tab == 'tab-squeezing':
            fig = create_squeezing_plot(freq_min, freq_max, xlen)
        else:
            fig = go.Figure()

    return dcc.Graph(figure=fig, style={'height': '600px'})

@app.callback(
    [Output('mm-ifo-omc-loss-display', 'children'),
     Output('mm-sqz-omc-loss-display', 'children')],
    [Input('mm-ifo-omc-input', 'value'),
     Input('mm-sqz-omc-input', 'value')]
)
def update_mm_loss_display(mm_ifo, mm_sqz):
    def fmt(val):
        if val is None or val == 0.0:
            return ''
        return f'→ {val**2 / 2 * 100:.3f}% implied power loss'
    return fmt(mm_ifo), fmt(mm_sqz)


def create_overview_plot(dataset, freq_min, freq_max, xlen=1000):
    """
    DARM ASD (left) + squeezing residuals (right).
    Mirrors notebook: meas_dict[meas][sqz_config], tab20 colors, no mutation.
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=('DARM Amplitude Spectral Density', 'Squeezing Residuals'),
        horizontal_spacing=0.12
    )

    # ── Unwrap nested structure: dataset = {meas_name: {sqz_config: {data}}} ──
    # dataset_name is the single top-level key loaded by fetch callback
    dataset_name = next(iter(dataset))
    inner        = dataset[dataset_name]

    darm_chan = 'GDS meters'

    # ── Build sqz_keys and tab20 colors exactly as notebook does ─────────────
    skip     = SKIP_KEYS | {'NLG'}
    sqz_keys = [k for k in inner
                if k not in skip
                and isinstance(inner[k], dict)
                and 'freq' in inner[k]]

    cmap   = cm.tab20
    n      = max(len(sqz_keys) - 1, 1)
    colors = {k: f'rgba{tuple(int(c*255) for c in cmap(i/n)[:3]) + (1,)}'
              for i, k in enumerate(sqz_keys)}
    colors['nosqz'] = 'black'

    nosqz_psd = inner.get('nosqz', {}).get(darm_chan + ' psd')

    for sqz_config in sqz_keys:
        entry = inner[sqz_config]
        freq  = np.array(entry['freq'])
        psd   = np.array(entry[darm_chan + ' psd'])

        # Rebin — always log-spaced, matches notebook rebin()
        freq_r, psd_r = rebin(freq, psd, xlen=xlen)
        asd_r         = psd_r ** 0.5
        color         = colors[sqz_config]

        # ── Label: GPS timestring + config name, exactly as notebook ──────────
        try:
            start_str, stop_str = get_gps_timestrings(entry)
            label = f"{start_str[:-3]}-{stop_str[:-7]+stop_str[8:-4]}, {sqz_config}"
        except Exception:
            label = str(sqz_config)

        # ── Left panel: DARM ASD ──────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=freq_r, y=asd_r,
            name=label,
            mode='lines',
            line=dict(color=color, width=0.8),
            opacity=0.85,
            legendgroup=sqz_config,
        ), row=1, col=1)

        # ── Right panel: squeezing residuals (skip nosqz) ─────────────────────
        if sqz_config != 'nosqz' and nosqz_psd is not None:
            nosqz_psd_r = rebin(freq, np.array(nosqz_psd), xlen=xlen)[1]
            residuals   = 10 * np.log10(psd_r / nosqz_psd_r)
            db          = estimate_median(freq_r, residuals, fcenter=1700, fband=200)[0]

            fig.add_trace(go.Scatter(
                x=freq_r, y=residuals,
                name=f"{db:.1f} dB",
                mode='lines',
                line=dict(color=color, width=0.8),
                opacity=0.85,
                legendgroup=sqz_config,
                showlegend=True,
            ), row=1, col=2)

    # ── Axes — match notebook xlim/ylim exactly ───────────────────────────────
    fig.update_xaxes(type='log',
                     range=[np.log10(freq_min), np.log10(freq_max)],
                     title_text='Frequency [Hz]', row=1, col=1)
    fig.update_yaxes(type='log',
                     range=[np.log10(0.5e-20), np.log10(3e-19)],
                     title_text='Displacement [m/√Hz]', row=1, col=1)

    fig.update_xaxes(type='log',
                     range=[np.log10(freq_min), np.log10(freq_max)],
                     title_text='Frequency [Hz]', row=1, col=2)
    fig.update_yaxes(type='linear',
                     range=[-6, 17],
                     tickvals=list(range(-6, 18, 2)),
                     title_text='dB [sqz / unsqz]', row=1, col=2)

    # Zero line on residuals panel (notebook: a4.axhline(0))
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'), row=1, col=2)

    fig.update_layout(
        height=550,
        template='plotly_white',
        font=dict(family='Arial', size=12, color=COLORS['text']),
        showlegend=True,
        hovermode='x unified',
        title=dict(text=f'{dataset_name}  —  DARM ASD', x=0.5, font=dict(size=14)),
        legend=dict(
            font=dict(size=9),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1,
        )
    )

    return fig


def create_noise_budget_plot(freq_min, freq_max, xlen=1000):
    """
    Noise budget plot mirroring notebook Step 7.
    Reads entirely from _cached_traces — no arguments needed beyond display params.
    """
    global _cached_traces

    if not _cached_traces:
        return go.Figure().add_annotation(
            text="Run analysis to generate noise budget",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['text_secondary'])
        )

    # ── Unpack cache ──────────────────────────────────────────────────────────
    dataset_name  = _cached_traces['dataset_name']
    freq          = _cached_traces['freq']
    traces_nosqz  = _cached_traces['traces_nosqz']
    ifo           = _cached_traces['ifo']
    darm_chan      = _cached_traces['darm_chan']
    sqz_keys       = _cached_traces['sqz_keys']
    colors         = _cached_traces['colors']
    inner          = _cached_traces['meas_dict'][dataset_name]

    # ── Best-squeezing config (notebook: FDS if present, else min median ASD) ─
    sqz_key = 'FDS' if 'FDS' in sqz_keys else \
              min(sqz_keys, key=lambda k: np.median(inner[k][darm_chan]))

    # ── Rebin measured data ───────────────────────────────────────────────────
    freq_r, nosqz_asd_r = rebin(freq, inner['nosqz'][darm_chan], xlen=xlen)
    _,      best_asd_r  = rebin(freq, inner[sqz_key][darm_chan], xlen=xlen)

    # ── Technical noise — already computed by run_analysis, stored on nosqz ──
    tn_psd     = np.array(inner['nosqz']['tn psd'])
    tn_psd_err = np.array(inner['nosqz']['tn psd err'])
    tn_asd     = np.nan_to_num(np.sqrt(np.abs(tn_psd)))
    tn_asd_err = np.nan_to_num(
        np.sqrt(np.abs(tn_psd + tn_psd_err)) - np.sqrt(np.abs(tn_psd))
    )
    freq_tn_r, tn_asd_r = rebin(freq, tn_asd, xlen=xlen)

    # ── Inferred quantum noise for best-sqz config ────────────────────────────
    freq_qn_r, sqz_qn_asd_r = rebin(
        freq, inner[sqz_key]['sqz qn asd'], xlen=xlen
    )

    # ── IFO state string for title (matches notebook ifo_str) ─────────────────
    hd_deg    = (ifo.Optics.Quadrature.dc - np.pi/2) * 180/np.pi
    eta       = ifo.Optics.PhotoDetectorEfficiency
    fc_det    = ifo.Squeezer.FilterCavity.fdetune
    parm_kW   = ifo.Laser.ArmPower / 1e3
    try:
        start_str, stop_str = get_gps_timestrings(inner['nosqz'])
        time_str = f"{start_str}–{stop_str}"
    except Exception:
        time_str = dataset_name
    ifo_str = (f"Parm={parm_kW:.0f}kW, HD={hd_deg:.1f}°, "
               f"η={eta:.2f}, fc={fc_det:.0f}Hz")

    # ── tn/sn ratio at fcenter ────────────────────────────────────────────────
    nosqz_lossy_qn_model_psd = traces_nosqz['Quantum'].psd
    tn_below_sn_dB, _, _ = estimate_median(
        freq,
        10 * np.log10(tn_psd / nosqz_lossy_qn_model_psd),
        fcenter=1300, fband=100
    )

    fig = go.Figure()

    # ── Measured nosqz DARM (notebook: red errorbar) ──────────────────────────
    fig.add_trace(go.Scatter(
        x=freq_r, y=nosqz_asd_r,
        name='Unsqz meas darm',
        mode='lines', line=dict(color='red', width=1.5),
        legendgroup='measured'
    ))

    # ── Best-sqz measured DARM (notebook: purple, alpha=0.5) ─────────────────
    fig.add_trace(go.Scatter(
        x=freq_r, y=best_asd_r,
        name=f'{sqz_key} meas darm',
        mode='lines', line=dict(color='purple', width=1.5),
        opacity=0.5,
        legendgroup='measured'
    ))

    # ── Lossless gwinc model (notebook: black dashed) ─────────────────────────
    # traces_nosqz was run with lossless readout first — stored as traces_noloss
    # in the cache if we add it; for now use the total asd from lossless run
    # NOTE: run_analysis stores traces_nosqz as the LOSSY run.
    
    fig.add_trace(go.Scatter(
        x=freq, y=traces_nosqz.asd,
        name=f'no sqz calc darm, lossy readout (η={eta:.2f})',
        mode='lines', line=dict(color='black', width=2),
        legendgroup='gwinc'
    ))

    nosqz_calc_asd = _cached_traces['nosqz_calc_asd']
    fig.add_trace(go.Scatter(
        x=freq, y=nosqz_calc_asd,
        name='no sqz calc darm, lossless readout',
        mode='lines', line=dict(color='black', width=1.5, dash='dash'),
        opacity=0.5, legendgroup='gwinc'
    ))

    # ── Technical noise (notebook: grey dotted errorbar) ─────────────────────
    fig.add_trace(go.Scatter(
        x=freq_tn_r, y=tn_asd_r,
        name='non-quantum technical noise (nosqz meas − lossy qn model)',
        mode='lines', line=dict(color='grey', width=1.5, dash='dot'),
        opacity=0.7,
        legendgroup='derived'
    ))

    # ── gwinc sub-budgets (notebook loop over CoatingBrownian, Quantum) ───────
    for trace_name in ['CoatingBrownian', 'Quantum']:
        try:
            trace_obj = traces_nosqz[trace_name]
            color = 'red' if trace_name == 'CoatingBrownian' else 'blue'
            fig.add_trace(go.Scatter(
                x=freq, y=trace_obj.asd,
                name=f'gwinc [{trace_name}]',
                mode='lines', line=dict(color=color, width=1.5),
                opacity=0.75,
                legendgroup='gwinc'
            ))
            # Quantum sub-traces (notebook: RelASSqz, RelASMisrotation, Readout)
            if trace_name == 'Quantum':
                for sub in ['RelASSqz', 'RelASMisrotation', 'Readout']:
                    try:
                        sub_obj = traces_nosqz['Quantum'][sub]
                        fig.add_trace(go.Scatter(
                            x=freq, y=sub_obj.asd,
                            name=f'gwinc [Quantum, {sub}]',
                            mode='lines',
                            line=dict(width=1.2, dash='dash'),
                            opacity=0.8,
                            legendgroup='gwinc'
                        ))
                    except Exception:
                        pass
        except Exception:
            pass

    # ── Inferred quantum noise for best-sqz config (notebook: lightpink dotted)
    fig.add_trace(go.Scatter(
        x=freq_qn_r, y=sqz_qn_asd_r,
        name=f'{sqz_key} inferred quantum noise asd',
        mode='lines', line=dict(color='lightpink', width=1.5, dash='dot'),
        opacity=0.7,
        legendgroup='derived'
    ))

    # ── Axes — match notebook xlim/ylim ──────────────────────────────────────
    fig.update_xaxes(
        type='log',
        range=[np.log10(freq_min), np.log10(freq_max)],
        title_text='Frequency [Hz]'
    )
    fig.update_yaxes(
        type='log',
        range=[np.log10(5e-21), np.log10(2e-19)],
        title_text='Displacement [m/√Hz]'
    )
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=True,
        hovermode='x',
        title=dict(
            text=(f'{dataset_name} — nosqz noise budget, {time_str}<br>'
                  f'{ifo_str}, tn/sn ~ {tn_below_sn_dB:.1f} dB'),
            x=0.5, font=dict(size=13)
        ),
        legend=dict(
            font=dict(size=9),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1,
            x=1.01, y=1, xanchor='left', yanchor='top'
        )
    )

    return fig


def create_squeezing_plot(freq_min, freq_max, xlen=1000):
    """
    Squeezing angle fit plot mirroring notebook per-dataset loop.
    - Uses budget (not budget_relgamma) for fitting — fitter makes internal copy
    - Uses auto_guess_deg anchored to sqz_min_rad from cache
    - Fresh QuantumRelGamma budget each iteration to avoid stale state
    - Stores reduced_dict back into _cached_traces for freq-dep plot to use
    """
    global _cached_traces

    if not _cached_traces:
        return go.Figure().add_annotation(
            text="Run analysis to generate squeezing plot",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['text_secondary'])
        )

    # ── Unpack cache ──────────────────────────────────────────────────────────
    dataset_name = _cached_traces['dataset_name']
    inner        = _cached_traces['meas_dict'][dataset_name]
    budget       = deepcopy(_cached_traces['budget'])
    ifo          = deepcopy(_cached_traces['ifo'])
    sqz_keys     = _cached_traces['sqz_keys']
    colors       = _cached_traces['colors']
    sqz_type     = _cached_traces['sqz_type']
    sqz_min_rad  = _cached_traces['sqz_min_rad']
    darm_chan     = _cached_traces['darm_chan']
    fcenter       = 1300

    # ── homodyne angle for axis label (matches notebook hd_ang_deg) ──────────
    hd_ang_deg  = (ifo.Optics.Quadrature.dc - np.pi/2) * 180/np.pi
    sqz_min_deg = np.rad2deg(sqz_min_rad)

    fig          = go.Figure()
    reduced_dict = {}

    for sqz_config in sqz_keys:
        if sqz_config == 'nosqz':
            continue

        freq        = np.array(inner[sqz_config]['freq'])
        sqz_qnr_dB = np.array(inner[sqz_config]['sqz_dB'])

        if sqz_qnr_dB is None:
            print(f"Warning: sqz_dB not found for {sqz_config}, skipping")
            continue

        # ── Rebin — always log-spaced, matches notebook rebin() ───────────────
        freq_r, dB_r = rebin(freq, sqz_qnr_dB, xlen=xlen)

        # ── Initial angle guess via auto_guess_deg (replaces sqz_angs_g0) ─────
        guess_deg = auto_guess_deg(sqz_config, sqz_min_deg)

        # ── Fit: pass budget (not budget_relgamma), fitter makes internal copy ─
        # This matches notebook: fit_sqz_dB(budget, ..., sqz_type=sqz_type)
        try:
            sqz_ang_rad = fit_sqz_dB(
                budget,
                meas_ff=freq_r,
                meas_dB=dB_r,
                sqz_type=sqz_type,
                guess=guess_deg,
                fmin=30,
                fmax=3500
            )
        except Exception as e:
            print(f"Fitting failed for {sqz_config}: {e}")
            sqz_ang_rad = np.deg2rad(guess_deg)

        sqz_ang_deg = np.rad2deg(sqz_ang_rad)

        # ── Fresh QuantumRelGamma budget each iteration (notebook pattern) ────
        ifo_local               = deepcopy(ifo)
        ifo_local.Squeezer.Type = sqz_type
        ifo_local.Squeezer.SQZAngle = sqz_ang_rad
        _budget_plot             = gwinc.load_budget('Aplus', bname='QuantumRelGamma')
        _budget_plot.ifo         = ifo_local
        traces_relgamma          = _budget_plot.run(freq=freq_r)
        fit_dB_data              = 10 * np.log10(traces_relgamma.psd)

        color      = colors.get(sqz_config, 'grey')
        median_sqz, _, _ = estimate_median(freq_r, dB_r, fcenter=fcenter, fband=200)

        # ── Measured data — thin line + alpha (notebook: lw=0.8, alpha=0.85) ──
        fig.add_trace(go.Scatter(
            x=freq_r, y=dB_r,
            name=f'{sqz_config}, @ {median_sqz:.2f} dB',
            mode='lines',
            line=dict(color=color, width=0.8),
            opacity=0.85,
            legendgroup=sqz_config,
        ))

        # ── Fitted model — thick dashed (notebook: lw=2, alpha=0.7, ls='--') ─
        fig.add_trace(go.Scatter(
            x=freq_r, y=fit_dB_data,
            name=f'{sqz_config} model @ {sqz_ang_deg + hd_ang_deg:.0f} deg',
            mode='lines',
            line=dict(color=color, width=2, dash='dash'),
            opacity=0.7,
            legendgroup=sqz_config,
        ))

        # ── Store reduced_dict entry (notebook pattern, all keys) ─────────────
        reduced_dict[sqz_config] = {
            'freq':              freq_r,
            'sqz_dB':            dB_r,
            'fit_sqz_dB':        fit_dB_data,
            'median_sqz_dB':     median_sqz,
            'color':             color,
            'sqz_ang_deg':       sqz_ang_deg + hd_ang_deg,
            'sqz_ang_gwinc_rad': sqz_ang_rad,
            'sqz_type':          sqz_type,
        }

    # ── Store reduced_dict and budget snapshot in cache for freq-dep plot ─────
    # Mirrors: meas_dict[meas]['reduced'] = reduced_dict
    reduced_dict['_budget']      = deepcopy(budget)
    reduced_dict['_sqz_type']    = sqz_type
    reduced_dict['_sqz_min_deg'] = sqz_min_deg
    _cached_traces['reduced_dict'] = reduced_dict

    # ── Vertical line at fcenter (notebook: ax.axvline) ───────────────────────
    fig.add_vline(x=fcenter, line=dict(color='grey', width=1, dash='dot'))

    # ── Title matches notebook ────────────────────────────────────────────────
    eta = ifo.Optics.PhotoDetectorEfficiency
    title_str = (f"{dataset_name}<br>"
                 f"Parm={ifo.Laser.ArmPower/1e3:.0f}kW, η={eta:.2f}")

    fig.update_xaxes(
        type='log',
        range=[np.log10(freq_min), np.log10(freq_max)],
        title_text='Frequency [Hz]'
    )
    fig.update_yaxes(
        range=[-7.5, 20],
        title_text='dB [sqz / unsqz]'
    )
    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=True,
        hovermode='x',
        title=dict(text=title_str, x=0.5, font=dict(size=13)),
        legend=dict(
            font=dict(size=8),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1,
            orientation='h',
            x=0.5, y=-0.15,
            xanchor='center', yanchor='top'
        )
    )

    return fig


# ============================================================================
# PROCESS / KINIT AND RUN THE GUI 
# ============================================================================

@app.callback(
    Output('info-panel', 'children'),
    Input('analysis-results-store', 'data')
)
def update_info_panel(analysis_results):
    """Display analysis information and metrics"""
    if not analysis_results:
        return html.Div(
            "Load data and run analysis to see metrics",
            style={'color': COLORS['text_secondary'], 'fontStyle': 'italic'}
        )

    global _cached_traces

    if not _cached_traces:
        return "No analysis data available"

    try:
        dataset_name = _cached_traces['dataset_name']
        inner        = _cached_traces['meas_dict'][dataset_name]
        sqz_keys     = _cached_traces['sqz_keys']
        ifo          = _cached_traces['ifo']

        # ── Extract all ifo state ─────────────────────────────────────────────
        eta         = ifo.Optics.PhotoDetectorEfficiency
        parm_kW     = ifo.Laser.ArmPower / 1e3
        hd_deg      = (ifo.Optics.Quadrature.dc - np.pi/2) * 180/np.pi
        fc_det      = ifo.Squeezer.FilterCavity.fdetune
        inj_loss    = ifo.Squeezer.InjectionLoss
        angle_rms   = ifo.Squeezer.SQZAngleRMS
        fc_L        = ifo.Squeezer.FilterCavity.L
        fc_Lrt      = ifo.Squeezer.FilterCavity.Lrt
        fc_Lrms     = ifo.Squeezer.FilterCavity.Lrms
        mm_ifo      = ifo.Optics.MM_IFO_OMC
        mm_ifo_phi  = ifo.Optics.MM_IFO_OMCphi
        mm_sqz      = ifo.Squeezer.MM_SQZ_OMC
        mm_sqz_phi  = ifo.Squeezer.MM_SQZ_OMCphi
        fc_l_mm     = ifo.Squeezer.FilterCavity.L_mm
        fc_psi_mm   = ifo.Squeezer.FilterCavity.psi_mm

        # ── Row helper ────────────────────────────────────────────────────────
        def row(label, value, note=''):
            return html.Div([
                html.Span(f"{label}: ",
                          style={'fontWeight': '500', 'fontSize': '12px',
                                 'width': '230px', 'display': 'inline-block'}),
                html.Span(value,
                          style={'fontFamily': 'monospace', 'fontSize': '11px'}),
                html.Span(f"  {note}" if note else '',
                          style={'color': COLORS['text_secondary'], 'fontSize': '11px'}),
            ], style={'marginLeft': '10px', 'marginTop': '2px'})

        info_sections = []

        # ── Status ────────────────────────────────────────────────────────────
        info_sections.append(html.Div([
            html.Strong("Analysis Status: "),
            html.Span("Complete ✓", style={'color': COLORS['success']})
        ]))
        info_sections.append(html.Hr(style={'margin': '8px 0'}))

        # ── Dataset and configs ───────────────────────────────────────────────
        info_sections.append(html.Strong(f"Dataset: {dataset_name}"))
        info_sections.append(html.Div(
            f"{len(sqz_keys)} configs: {', '.join(str(k) for k in sqz_keys)}",
            style={'marginLeft': '10px', 'marginTop': '3px', 'fontSize': '12px'}
        ))
        info_sections.append(html.Hr(style={'margin': '8px 0'}))

        # ── GPS times ─────────────────────────────────────────────────────────
        info_sections.append(html.Strong("GPS Time Ranges:"))
        for sqz_config in sqz_keys:
            entry = inner.get(sqz_config, {})
            if 'gps start' not in entry:
                continue
            try:
                start_str, stop_str = get_gps_timestrings(entry)
                time_label = f"{start_str} – {stop_str}"
            except Exception:
                time_label = f"{entry['gps start']:.0f} – {entry['gps stop']:.0f}"
            info_sections.append(html.Div([
                html.Span(f"{str(sqz_config)}: ",
                          style={'fontWeight': '500', 'fontSize': '12px'}),
                html.Span(time_label,
                          style={'fontFamily': 'monospace', 'fontSize': '11px'})
            ], style={'marginLeft': '10px', 'marginTop': '2px'}))
        info_sections.append(html.Hr(style={'margin': '8px 0'}))

        # ── Interferometer ────────────────────────────────────────────────────
        info_sections.append(html.Strong("Interferometer:"))
        info_sections += [
            row('Arm Power',         f'{parm_kW:.0f} kW'),
            row('Homodyne angle',    f'{hd_deg:.2f} deg'),
            row('FC detuning',       f'{fc_det:.0f} Hz'),
        ]
        info_sections.append(html.Hr(style={'margin': '8px 0'}))

        # ── Loss Budget ───────────────────────────────────────────────────────
        info_sections.append(html.Strong("Loss Budget:"))
        info_sections += [
            row('Readout efficiency η',
                f'{eta:.4f}',
                f'{(1-eta)*100:.2f}% total readout loss'),
            row('Injection loss',
                f'{inj_loss:.4f}',
                f'{inj_loss*100:.2f}%'),
            row('SQZ angle RMS',
                f'{angle_rms*1e3:.1f} mrad'),
            row('Implied total η',
                f'{eta * (1 - inj_loss):.4f}'),
        ]
        info_sections.append(html.Hr(style={'margin': '8px 0'}))

        # ── Filter Cavity ─────────────────────────────────────────────────────
        info_sections.append(html.Strong("Filter Cavity:"))
        info_sections += [
            row('FC length',             f'{fc_L:.2f} m'),
            row('FC round-trip loss',    f'{fc_Lrt*1e6:.1f} ppm'),
            row('FC length noise RMS',   f'{fc_Lrms:.2e} m'),
        ]
        info_sections.append(html.Hr(style={'margin': '8px 0'}))

        # ── Mode Mismatch ─────────────────────────────────────────────────────
        info_sections.append(html.Strong("Mode Mismatch:"))
        info_sections += [
            row('MM IFO→OMC amplitude',
                f'{mm_ifo:.4f}',
                f'{mm_ifo**2/2*100:.3f}% implied power loss'),
            row('MM IFO→OMC phase',
                f'{np.rad2deg(mm_ifo_phi):.2f} deg'),
            row('MM sqz→OMC amplitude',
                f'{mm_sqz:.4f}',
                f'{mm_sqz**2/2*100:.3f}% implied power loss'),
            row('MM sqz→OMC phase',
                f'{np.rad2deg(mm_sqz_phi):.2f} deg'),
            row('FC length mismatch',
                f'{fc_l_mm:.4f} m'),
            row('FC Gouy phase mismatch',
                f'{np.rad2deg(fc_psi_mm):.2f} deg'),
        ]
        info_sections.append(html.Hr(style={'margin': '8px 0'}))

        # ── Fitted squeezing angles (populated after squeezing tab visited) ───
        reduced_dict = _cached_traces.get('reduced_dict', {})
        if reduced_dict:
            info_sections.append(html.Strong("Fitted Squeezing Angles:"))
            for sqz_config in sqz_keys:
                if sqz_config in reduced_dict and 'sqz_ang_deg' in reduced_dict[sqz_config]:
                    ang = reduced_dict[sqz_config]['sqz_ang_deg']
                    med = reduced_dict[sqz_config]['median_sqz_dB']
                    info_sections.append(html.Div([
                        html.Span(f"{str(sqz_config)}: ",
                                  style={'fontWeight': '500', 'fontSize': '12px'}),
                        html.Span(f"{ang:.1f}° fitted,  {med:.2f} dB @ 1300 Hz",
                                  style={'fontFamily': 'monospace', 'fontSize': '11px'})
                    ], style={'marginLeft': '10px', 'marginTop': '2px'}))

        return info_sections

    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div([
            html.Strong("Error: ", style={'color': COLORS['error']}),
            html.Span(str(e), style={'fontSize': '11px', 'fontFamily': 'monospace'})
        ])


@app.callback(
    [Output('kerberos-modal',    'is_open', allow_duplicate=True),
     Output('kerberos-status',   'children'),
     Output('kerberos-username', 'value'),
     Output('kerberos-password', 'value')],
    [Input('kerberos-submit', 'n_clicks'),
     Input('kerberos-cancel', 'n_clicks')],
    [State('kerberos-username', 'value'),
     State('kerberos-password', 'value'),
     State('kerberos-modal',    'is_open')],
    prevent_initial_call=True
)
def handle_kerberos_auth(submit_clicks, cancel_clicks, username, password, is_open):
    """Handle Kerberos authentication from modal"""
    ctx       = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if button_id == 'kerberos-cancel':
        return False, "", "", ""

    if button_id == 'kerberos-submit':
        if not username or not password:
            return True, "Please enter both username and password", username, password
        success, message = authenticate_kerberos(username, password)
        if success:
            return True, "✓ Authenticated! Click Load Data to fetch.", "", ""
        return True, f"Authentication failed: {message}", username, password

    return is_open, "", username, password


# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == '__main__':
    app.server.config['TIMEOUT'] = 600
    app.run(debug=True, port=8050, threaded=True)