import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# ==============================================================================
# ENGINE: PUBLICATION-QUALITY FIGURE GENERATOR (REAL DATA)
# ==============================================================================

# Configurações de estilo para MNRAS (Times New Roman, tamanhos de fonte acadêmicos)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['axes.titlesize'] = 12

WP_FIXED = 18.0803

def extract_exact(ra, z, res, wp):
    X = z * np.cos(ra - wp * z)
    Y = z * np.sin(ra - wp * z)
    A_matrix = np.vstack([X, Y]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, res, rcond=None)
    D0 = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
    phi0 = np.arctan2(coeffs[1], coeffs[0])
    if phi0 < 0: phi0 += 2*np.pi
    return D0, phi0

def extract_amplitude_frozen(ra, z, res, phi0, wp):
    V = z * np.cos(ra - (phi0 + wp * z))
    D0 = np.sum(V * res) / np.sum(V * V)
    res_mod = res - D0 * V
    sig = np.sqrt(np.sum(res_mod**2) / (len(res)-1))
    err = sig / np.sqrt(np.sum(V * V))
    return D0 / err

if __name__ == "__main__":
    print("========================================================================")
    print(" GERANDO FIGURAS PARA O ARTIGO (DADOS REAIS)")
    print("========================================================================")
    
    archive = np.load("purified_residuals.npz")
    ra = archive['ra']
    z = archive['z']
    res = archive['res']
    n_obj = len(ra)
    
    # ----------------------------------------------------------------------
    # FIGURA 2: MONTE CARLO NULL TESTING
    # ----------------------------------------------------------------------
    print("[1/3] Gerando Figura 2 (Monte Carlo Z-Score)...")
    np.random.seed(42)
    D0_real, _ = extract_exact(ra, z, res, WP_FIXED)
    
    null_amps = []
    for _ in range(1000):
        ra_shuffled = np.random.permutation(ra)
        D0_n, _ = extract_exact(ra_shuffled, z, res, WP_FIXED)
        null_amps.append(D0_n)
    
    mu_null = np.mean(null_amps)
    sig_null = np.std(null_amps)
    z_real = (D0_real - mu_null) / sig_null
    z_nulls = (np.array(null_amps) - mu_null) / sig_null
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z_nulls, bins=30, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Monte Carlo Null Distribution (1,000 Shuffles)')
    ax.axvline(x=z_real, color='crimson', linewidth=2, label='Raw SDSS Detection')
    
    ax.annotate(f'Z-score Significance: {z_real:.2f}σ', xy=(z_real, 0.2), xytext=(z_real - 8, 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, fontweight='bold', ha='center')
    
    ax.set_xlabel('Z-score Significance (σ)')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(-4, 20)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('fig2_montecarlo.pdf', dpi=300)
    plt.close()
    
    # ----------------------------------------------------------------------
    # FIGURA 3: SPLIT-HALF CROSS-VALIDATION
    # ----------------------------------------------------------------------
    print("[2/3] Gerando Figura 3 (Split-Half)...")
    np.random.seed(42)
    indices = np.arange(n_obj)
    
    phi_A_list, phi_B_list, sig_B_list = [], [], []
    for _ in range(100):
        np.random.shuffle(indices)
        idx_A = indices[:n_obj//2]
        idx_B = indices[n_obj//2:]
        
        _, phi_A = extract_exact(ra[idx_A], z[idx_A], res[idx_A], WP_FIXED)
        _, phi_B = extract_exact(ra[idx_B], z[idx_B], res[idx_B], WP_FIXED)
        sig_B = extract_amplitude_frozen(ra[idx_B], z[idx_B], res[idx_B], phi_A, WP_FIXED)
        
        phi_A_list.append(np.degrees(phi_A))
        phi_B_list.append(np.degrees(phi_B))
        sig_B_list.append(sig_B)
        
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    
    # Painel Esquerdo: Consistência da Fase
    ax0 = plt.subplot(gs[0])
    ax0.scatter(phi_A_list, phi_B_list, alpha=0.7, color='peru', edgecolor='black')
    min_p, max_p = min(phi_A_list), max(phi_A_list)
    ax0.plot([min_p, max_p], [min_p, max_p], 'k--', label='1:1 Consistency')
    ax0.set_xlabel(r'Phase $\phi_0$ (Training Half A) [deg]')
    ax0.set_ylabel(r'Phase $\phi_0$ (Validation Half B) [deg]')
    ax0.set_title('Transversal Phase Consistency')
    ax0.legend()
    
    # Painel Direito: Histograma de Significância
    ax1 = plt.subplot(gs[1])
    ax1.hist(sig_B_list, bins=15, color='seagreen', alpha=0.7, edgecolor='black')
    med_sig = np.median(sig_B_list)
    ax1.axvline(x=5.0, color='crimson', linestyle='--', linewidth=2, label='Discovery Threshold (5σ)')
    ax1.axvline(x=med_sig, color='navy', linewidth=2, label=f'Median: {med_sig:.2f}σ')
    ax1.set_xlabel(r'Detection Significance $D_0$ on Unseen Data ($\sigma$)')
    ax1.set_ylabel('Frequency (100 Splits)')
    ax1.set_title('Cross-Validation Performance')
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('fig3_splithalf.pdf', dpi=300)
    plt.close()

    # ----------------------------------------------------------------------
    # FIGURA 4: ALINHAMENTO DO CMB (MOLLWEIDE)
    # ----------------------------------------------------------------------
    print("[3/3] Gerando Figura 4 (Mollweide CMB Alignment)...")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='mollweide')
    
    # SDSS Real Phase (extração matemática real a partir dos resíduos purificados)
    _, phi_global_rad = extract_exact(ra, z, res, WP_FIXED)
    sdss_ra = phi_global_rad
    if sdss_ra > np.pi: sdss_ra -= 2*np.pi
    
    # Planck Axis of Evil (approx l=-100, b=60 -> RA ~ 168 deg = 2.93 rad)
    cmb_ra = 168.0 * np.pi / 180.0
    if cmb_ra > np.pi: cmb_ra -= 2*np.pi
    
    ax.scatter(sdss_ra, 0, marker='*', s=300, color='crimson', edgecolor='black', label=rf'SDSS DR16Q Phase ($\phi_0={np.degrees(phi_global_rad):.1f}^\circ$)')
    ax.scatter(cmb_ra, np.deg2rad(15), marker='X', s=200, color='navy', label='CMB "Axis of Evil" Anomaly')
    
    ax.annotate(r'$\omega_p = 18.08$ rad/z', xy=(sdss_ra + 0.3, 0.3), xytext=(sdss_ra + 0.8, 0.6),
            arrowprops=dict(facecolor='goldenrod', shrink=0.05, width=2), fontsize=12, color='darkgoldenrod')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title('Transversal Phase Alignment: SDSS vs CMB', pad=20)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.tight_layout()
    plt.savefig('fig4_cmb_alignment.pdf', dpi=300)
    plt.close()
    
    print("\n[SUCESSO] Figuras reais exportadas: fig2_montecarlo.pdf, fig3_splithalf.pdf, fig4_cmb_alignment.pdf")