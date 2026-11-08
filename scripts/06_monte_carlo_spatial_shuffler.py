import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ENGINE: MONTE CARLO NULL TESTING (EXACT LINEAR SHUFFLER)
# ==============================================================================

WP_FIXED = 18.0803  # Prior do Pantheon+
N_SHUFFLES = 1000

def extract_exact_amplitude(ra, z, res, wp):
    X = z * np.cos(ra - wp * z)
    Y = z * np.sin(ra - wp * z)
    A_matrix = np.vstack([X, Y]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, res, rcond=None)
    D0 = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
    return D0

if __name__ == "__main__":
    print("========================================================================")
    print(f" EXECUTANDO: MONTE CARLO ESTOCÁSTICO REALS ({N_SHUFFLES} ITERAÇÕES)")
    print("========================================================================")
    
    # 1. Carregar dados limpos do script anterior
    try:
        archive = np.load("purified_residuals.npz")
        ra_real = archive['ra']
        z_real = archive['z']
        res_real = archive['res']
    except FileNotFoundError:
        print("[ERRO] Você precisa rodar o script '05_orthogonal_systematics_cleaning.py' primeiro!")
        exit()
        
    print(f"[1/3] Extraindo o sinal real do céu limpo...")
    D0_real = extract_exact_amplitude(ra_real, z_real, res_real, WP_FIXED)
    print(f"      Amplitude Bruta Detectada (D₀): {D0_real:.5f}")
    
    # 2. Rodar o loop de Monte Carlo
    print(f"[2/3] Rodando {N_SHUFFLES} embaralhamentos de coordenadas...")
    null_amplitudes = []
    
    # Fixar a semente aleatória apenas para reprodutibilidade estrita do teste
    np.random.seed(42)
    
    for i in range(N_SHUFFLES):
        # Embaralha apenas a Ascensão Reta, quebrando a coerência direcional espacial
        ra_shuffled = np.random.permutation(ra_real)
        D0_null = extract_exact_amplitude(ra_shuffled, z_real, res_real, WP_FIXED)
        null_amplitudes.append(D0_null)
        
        if (i + 1) % 100 == 0:
            print(f"      Progresso: {i + 1}/{N_SHUFFLES} simulações concluídas...")
            
    null_amplitudes = np.array(null_amplitudes)
    
    # 3. Análise Estatística Rígida
    print("[3/3] Calculando métricas de background...")
    mu_null = np.mean(null_amplitudes)
    sigma_null = np.std(null_amplitudes)
    max_null = np.max(null_amplitudes)
    
    # Z-score real do sinal contra o fundo estocástico real
    z_score_real = (D0_real - mu_null) / sigma_null
    margin_above_max = D0_real / max_null
    
    print("\n========================================================================")
    print(" RESULTADO REAL DO TESTE DE MONTE CARLO")
    print("========================================================================")
    print(f" Média do Fundo de Ruído (μ_null)       : Amplitude = {mu_null:.5f}")
    print(f" Desvio Padrão do Fundo (σ_null)        : {sigma_null:.5f}")
    print(f" Maior Falso Positivo Gerado (Max_null) : Amplitude = {max_null:.5f}")
    print("------------------------------------------------------------------------")
    print(f" TRUE Z-SCORE NORMALIZED SIGNIFICANCE   : {z_score_real:.2f}σ")
    print(f" Margem sobre o maior ruído empírico    : {margin_above_max:.2f}x")
    print("========================================================================\n")