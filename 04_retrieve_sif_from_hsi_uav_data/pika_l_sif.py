import os
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid as trapz  # 使用新版 scipy 的 trapezoid
# (在 from scipy.integrate import trapezoid as trapz 附近)
from scipy.optimize import curve_fit
import warnings

# ----------------------------------------------------------------------
# --- 1. 配置参数 ---
# ----------------------------------------------------------------------

# --- 路径 ---
REFLECTANCE_BASE_DIR = '/pg_caches/HSI_cache/reflectance_csv'
IRRADIANCE_BASE_DIR = '/pg_caches/HSI_cache/USB2K'
OUTPUT_CSV_PATH = '/pg_caches/HSI_cache/reflectance_csv/sif_and_vi_results.csv'

# --- 常量 ---
GLOBAL_CF = 7.3562e-04  # (W·m⁻²·nm⁻¹) / Counts

# --- 高光谱波段 (来自您的列表) ---
HSI_WAVELENGTHS = pd.read_csv('/pg_caches/HSI_cache/wavelength.csv', header=None).squeeze().values

# --- SIF 和 VI 波段定义 (nm) ---
SIF_BANDS_TARGETS = {
    'r_fld_in': 687,   'r_fld_out': 680,
    'a_fld_in': 761,   'a_fld_out': 754,
    'a_3fld_l': 750,   'a_3fld_in': 762,  'a_3fld_r': 777,
}
VI_BANDS_TARGETS = {
    'red': 670,   # 用于 NDVI, EVI2, NIRv
    'nir': 800,   # 用于 NDVI, EVI2, NIRv
    'blue': 470   # 用于 EVI2 (虽然 EVI2 公式不需要, 但 EVI 需要)
}

# --- 步骤 2: HSI 和 USB2000 的对应关系 (手动解析) ---
# (已自动修正您在 ls -la 中 'AbsoluteIrradiance' 的拼写错误)
DATA_MAPPING = {
    '2023_08_04_09_44_30': {
        'irrad_dir': '20230804_0948_N_11',
        'files': ['AbsoluteIrradiance_09-52-23-270.txt', 'AbsoluteIrradiance_09-52-16-070.txt']
    },
    '2023_08_04_15_33_12': {
        'irrad_dir': '20230804_1537_N_15',
        'files': ['AbsoluteIrradiance_15-40-31-476.txt', 'AbsoluteIrradiance_15-37-25-794.txt']
    },
    '2023_08_04_17_57_02': {
        'irrad_dir': '20230804_1759_N_12',
        'files': ['AbsoluteIrradiance_18-02-20-981.txt', 'AbsoluteIrradiance_17-59-59-557.txt', 'AbsoluteIrradiance_18-02-23-455.txt']
    },
    '2023_08_09_15_49_08': {
        'irrad_dir': '20230809_1552_N_11',
        'files': ['AbsoluteIrradiance_15-53-48-433.txt', 'AbsoluteIrradiance_15-52-47-977.txt', 'AbsoluteIrradiance_15-52-46-090.txt']
    },
    '2023_08_09_16_46_06': {
        'irrad_dir': '20230809_1650_N_9',
        'files': ['AbsoluteIrradiance_16-50-24-345.txt', 'AbsoluteIrradiance_16-50-43-898.txt', 'AbsoluteIrradiance_16-51-51-421.txt']
    },
    '2023_08_09_17_44_34': {
        'irrad_dir': '20230809_1749_N_11',
        'files': ['AbsoluteIrradiance_17-55-50-907.txt', 'AbsoluteIrradiance_17-49-43-641.txt', 'AbsoluteIrradiance_17-55-44-553.txt']
    },
    '2023_08_09_18_46_12': {
        'irrad_dir': '20230809_1850_N_9',
        'files': ['AbsoluteIrradiance_18-50-29-177.txt', 'AbsoluteIrradiance_18-50-23-009.txt', 'AbsoluteIrradiance_18-50-16-741.txt']
    },
    '2023_08_09_19_37_16': {
        'irrad_dir': '20230809_1940_N_11',
        'files': ['AbsoluteIrradiance_19-40-48-270.txt', 'AbsoluteIrradiance_19-42-47-515.txt']
    },
    '2023_08_12_15_04_01': {
        'irrad_dir': '20230812_1508_N_12',
        'files': ['AbsoluteIrradiance_15-09-10-579.txt', 'AbsoluteIrradiance_15-20-52-685.txt', 'AbsoluteIrradiance_15-20-52-685.txt']
    }
    # ... 您可以按此格式添加更多 ...
}

# --- 新增：ROI 编号到处理名称的映射 ---
ROI_STRESS_MAP = {
    # 键是字符串，因为它们来自 CSV 的列名
    # 底行 (0-11)
    '0': 'W1S0', '1': 'W1S1', '2': 'W1S2', '3': 'W1S3', '4': 'W2S3', '5': 'W2S2', '6': 'W2S1', '7': 'W2S0', '8': 'W3S0', '9': 'W3S2', '10': 'W3S3', '11': 'W3S1',
    # 中行 (12-23)
    '12': 'W2S1', '13': 'W2S2', '14': 'W2S3', '15': 'W2S0', '16': 'W1S2', '17': 'W1S3', '18': 'W1S1', '19': 'W1S0', '20': 'W3S0', '21': 'W3S1', '22': 'W3S3', '23': 'W3S2',
    # 顶行 (24-35)
    '24': 'W2S0', '25': 'W2S2', '26': 'W2S3', '27': 'W2S1', '28': 'W3S3', '29': 'W3S2', '30': 'W3S1', '31': 'W3S0', '32': 'W1S2', '33': 'W1S3', '34': 'W1S1', '35': 'W1S0',
    # 补充 (36)
    '36': 'W0S0'
}
# --- 结束新增 ---

# ----------------------------------------------------------------------
# --- 3. 辅助函数 ---
# ----------------------------------------------------------------------

def parse_spectrum(filepath):
    """解析USB2000的txt文件"""
    encodings = ['gb18030', 'gbk', 'utf-8']
    header_line_num = -1
    correct_encoding = None
    try:
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    for i, line in enumerate(f):
                        if '>>>>>Begin Spectral Data<<<<<' in line:
                            header_line_num = i
                            correct_encoding = encoding
                            break
                if header_line_num != -1: break
            except (UnicodeDecodeError, StopIteration):
                continue
        
        if header_line_num == -1:
            print(f"    [警告] 未能在 {filepath} 中找到光谱数据头。")
            return pd.DataFrame()

        df = pd.read_csv(
            filepath, sep='\t', skiprows=header_line_num + 1,
            header=None, names=['wavelength', 'intensity'],
            encoding=correct_encoding
        )
        df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce')
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        return df.dropna()
    except Exception as e:
        print(f"    [错误] 解析 {filepath} 失败: {e}")
        return pd.DataFrame()

def prepare_irradiance(irrad_file_paths, hsi_wavelengths, global_cf):
    """
    读取、平均、校准并插值辐照度数据。
    返回: (插值到HSI波段的辐照度, 入射PAR值 (W/m²))
    """
    dfs = []
    for f in irrad_file_paths:
        df = parse_spectrum(f)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        raise ValueError(f"在 {irrad_file_paths} 中未找到有效的辐照度文件。")
    
    # 1. 计算平均光谱 (Counts)
    df_concat = pd.concat(dfs)
    df_avg_counts = df_concat.groupby('wavelength')['intensity'].mean().reset_index()
    
    # 2. 校准为 W·m⁻²·nm⁻¹
    df_calibrated = df_avg_counts.copy()
    df_calibrated['irrad_wm2_nm'] = df_calibrated['intensity'] * global_cf
    
    # 3. 计算入射 PAR (W/m²)
    par_spectrum = df_calibrated[
        (df_calibrated['wavelength'] >= 400) & (df_calibrated['wavelength'] <= 700)
    ]
    incident_par_wm2 = trapz(par_spectrum['irrad_wm2_nm'], par_spectrum['wavelength'])
    
    # 4. 插值到 HSI 波段
    f_interp = interp1d(
        df_calibrated['wavelength'], 
        df_calibrated['irrad_wm2_nm'],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    interpolated_irradiance_hsi = f_interp(hsi_wavelengths)
    
    return interpolated_irradiance_hsi, incident_par_wm2

def find_band_index(wavelengths, target_wav):
    """找到最接近目标波长的索引"""
    return np.argmin(np.abs(wavelengths - target_wav))

# --- SIF 和 VI 计算公式 ---
# (使用 numpy 数组操作，忽略运行时警告)
def calc_sif_fld(L_in, L_out, I_in, I_out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return ((I_out * L_in) - (I_in * L_out)) / (I_out - I_in)

def calc_sif_3fld(L_l, L_in, L_r, I_l, I_in, I_r, wav_l, wav_in, wav_r):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        omega_l = (wav_r - wav_in) / (wav_r - wav_l)
        omega_r = (wav_in - wav_l) / (wav_r - wav_l)
        
        numerator = L_in * (omega_l * I_l + omega_r * I_r) - I_in * (omega_l * L_l + omega_r * L_r)
        denominator = (omega_l * I_l + omega_r * I_r) - I_in
        return numerator / denominator

def calc_ndvi(nir, red):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (nir - red) / (nir + red)

def calc_evi2(nir, red):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)

def calc_nirv(nir, red):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return calc_ndvi(nir, red) * nir
def sfm_model(lambda_array, a, b, c, irradiance_array):
    """
    SFM (Spectral Function Method) model (L = I*R + F).
    L = I * (a*lambda + b) + c * f(lambda)
    f(lambda) = exp(-((lambda - 740) / (21*sqrt(2)))^2)
    """
    # (21*sqrt(2))**2 = 882.0
    f_lambda = np.exp(-((lambda_array - 740.0)**2) / 882.0)
    
    # R(lambda) = a*lambda + b
    reflectance_model = a * lambda_array + b
    
    # L(lambda)
    radiance_model = irradiance_array * reflectance_model + c * f_lambda
    return radiance_model
# ----------------------------------------------------------------------
# --- 4. 主执行函数 ---
# ----------------------------------------------------------------------

def main():
    print("=== 开始批量 SIF 和 VI 计算 ===")
    
    all_results = [] # 存储所有结果行

    # --- 预先计算 HSI 波段索引 ---
    print("正在计算 HSI 波段索引...")
    sif_indices = {name: find_band_index(HSI_WAVELENGTHS, wav) for name, wav in SIF_BANDS_TARGETS.items()}
    vi_indices = {name: find_band_index(HSI_WAVELENGTHS, wav) for name, wav in VI_BANDS_TARGETS.items()}
    print("  正在计算 SFM 拟合范围 (745-780 nm) 索引...")
    sfm_fit_indices = np.where((HSI_WAVELENGTHS >= 745) & (HSI_WAVELENGTHS <= 780))[0]
    lambda_sfm_subset = HSI_WAVELENGTHS[sfm_fit_indices]
    
    if len(lambda_sfm_subset) < 4: # 至少需要4个点来拟合3个参数
        print(f"  [警告] SFM 拟合波段不足 ({len(lambda_sfm_subset)}个)，SFM 将被禁用。")
        sfm_fit_indices = None
    else:
        print(f"  SFM 拟合将使用 {len(lambda_sfm_subset)} 个波段 (从 {lambda_sfm_subset[0]:.1f}nm 到 {lambda_sfm_subset[-1]:.1f}nm)。")

    print("  SIF 索引:", sif_indices)
    print("  VI 索引:", vi_indices)

    # 1. 遍历 DATA_MAPPING 中的每个 HSI 文件夹
    for hsi_dir_name, irrad_info in DATA_MAPPING.items():
        
        csv_path = os.path.join(REFLECTANCE_BASE_DIR, hsi_dir_name, 'all_pixels.csv')
        print(f"\n--- 正在处理: {csv_path} ---")
        
        if not os.path.exists(csv_path):
            print(f"  [警告] 文件未找到，跳过: {csv_path}")
            continue
            
        # 2. 解析日期和时间
        try:
            dt_parts = hsi_dir_name.split('_')
            date_str = f"{dt_parts[0]}-{dt_parts[1]}-{dt_parts[2]}"
            time_str = f"{dt_parts[3]}:{dt_parts[4]}:{dt_parts[5]}"
        except Exception:
            date_str, time_str = hsi_dir_name, hsi_dir_name # 回退

        # 3. 准备该 HSI 对应的辐照度
        print("  正在准备辐照度...")
        irrad_file_paths = [
            os.path.join(IRRADIANCE_BASE_DIR, irrad_info['irrad_dir'], f)
            for f in irrad_info['files']
        ]
        
        try:
            # irrad_hsi: 插值后的辐照度 (W·m⁻²·nm⁻¹) (150个波段)
            # par_wm2:   总入射 PAR (W/m²) (单个值)
            irrad_hsi, par_wm2 = prepare_irradiance(irrad_file_paths, HSI_WAVELENGTHS, GLOBAL_CF)
            print(f"  总入射 PAR (W/m²): {par_wm2:.4f}")
        except Exception as e:
            print(f"  [错误] 无法准备辐照度，跳过此文件: {e}")
            continue
            
        # 4. 读取反射率 CSV
        print(f"  读取反射率 CSV: {csv_path}")
        try:
            # 假设 CSV 的列是 ROI ('1', '2', ...)，行是波段
            ref_df = pd.read_csv(csv_path)
            # 将波长设置为索引，以便轻松访问
            ref_df.index = HSI_WAVELENGTHS
        except Exception as e:
            print(f"  [错误] 无法读取 CSV，跳过此文件: {e}")
            continue
            
        # 5. 循环处理 CSV 中的每一列 (每个 ROI)
        print(f"  开始计算 {len(ref_df.columns)} 个 ROIs...")
        for roi_col_name in ref_df.columns:
            
            # (N_bands,) 数组
            reflectance_roi = ref_df[roi_col_name].values
            
            # (N_bands,) 数组
            radiance_roi = reflectance_roi * irrad_hsi
            
            # --- 提取计算所需的单个值 ---
            
            # VIs
            ref_red = reflectance_roi[vi_indices['red']]
            ref_nir = reflectance_roi[vi_indices['nir']]
            
            # SIF (L = Radiance, I = Irradiance)
            L_687 = radiance_roi[sif_indices['r_fld_in']]
            L_680 = radiance_roi[sif_indices['r_fld_out']]
            I_687 = irrad_hsi[sif_indices['r_fld_in']]
            I_680 = irrad_hsi[sif_indices['r_fld_out']]
            
            L_761 = radiance_roi[sif_indices['a_fld_in']]
            L_754 = radiance_roi[sif_indices['a_fld_out']]
            I_761 = irrad_hsi[sif_indices['a_fld_in']]
            I_754 = irrad_hsi[sif_indices['a_fld_out']]

            L_750 = radiance_roi[sif_indices['a_3fld_l']]
            L_762 = radiance_roi[sif_indices['a_3fld_in']]
            L_777 = radiance_roi[sif_indices['a_3fld_r']]
            I_750 = irrad_hsi[sif_indices['a_3fld_l']]
            I_762 = irrad_hsi[sif_indices['a_3fld_in']]
            I_777 = irrad_hsi[sif_indices['a_3fld_r']]
            wav_750 = HSI_WAVELENGTHS[sif_indices['a_3fld_l']]
            wav_762 = HSI_WAVELENGTHS[sif_indices['a_3fld_in']]
            wav_777 = HSI_WAVELENGTHS[sif_indices['a_3fld_r']]

            # --- 执行计算 ---
            sif_687_fld = calc_sif_fld(L_687, L_680, I_687, I_680)
            sif_761_fld = calc_sif_fld(L_761, L_754, I_761, I_754)
            sif_3fld = calc_sif_3fld(
                L_750, L_762, L_777,
                I_750, I_762, I_777,
                wav_750, wav_762, wav_777
            )
            
            ndvi = calc_ndvi(ref_nir, ref_red)
            evi2 = calc_evi2(ref_nir, ref_red)
            nirv = calc_nirv(ref_nir, ref_red)
            # --- SFM 计算 ---
            sif_sfm = np.nan # 默认为 nan
            
            if sfm_fit_indices is not None:
                try:
                    # 提取拟合所需的数据子集
                    radiance_sfm_subset = radiance_roi[sfm_fit_indices]
                    irradiance_sfm_subset = irrad_hsi[sfm_fit_indices]
                    
                    # 运行非线性拟合: L = f(lambda, a, b, c, I)
                    popt, pcov = curve_fit(
                        lambda wl, a, b, c: sfm_model(wl, a, b, c, irradiance_sfm_subset),
                        lambda_sfm_subset,       # X_data
                        radiance_sfm_subset,     # Y_data
                        p0=[0, 0, 0],            # 初始猜测 (a, b, c)
                        maxfev=5000              # 增加迭代次数以保证收敛
                    )
                    sif_sfm = popt[2] # 'c' 是 SIF @ 740nm
                except RuntimeError:
                    # 拟合失败，sif_sfm 保持为 nan
                    pass 
                except Exception:
                    # 其他意外错误，sif_sfm 保持为 nan
                    pass
            # 6. 存储结果 (已修改)
            
            # --- 新增：查找处理名称 ---
            # 使用 .get() 安全地查找，如果CSV列名不在MAP中(例如 'wavelength')，则返回 'Unknown'
            stress_name = ROI_STRESS_MAP.get(str(roi_col_name), 'Unknown') 
            
            result_row = {
                'input_csv_path': csv_path,
                'date': date_str,
                'time_cst': time_str,
                'roi_column': roi_col_name,
                'stress_name': stress_name,  # <-- 新增列
                'sif_3fld': sif_3fld,
                'sif_687_fld': sif_687_fld,
                'sif_761_fld': sif_761_fld,
                'sif_sfm_740': sif_sfm, # <-- 新增 SIF
                'W': stress_name[0:2], # <-- 新增列 (例如 'W1')
                'S': stress_name[2:4], # <-- 新增列 (例如 'S0')
                'par_wm2': par_wm2,  # 这是总的入射 PAR
                'nirv': nirv,
                'evi2': evi2,
                'ndvi': ndvi,
                'sif_3fld_yield': sif_761_fld/par_wm2,
                'sif_687_fld_yield': sif_687_fld/par_wm2,
                'sif_761_fld_yield': sif_761_fld/par_wm2,
                'sif_sfm_740_yield': sif_sfm/par_wm2,
            }
            
            # 仅在 'stress_name' 不是 'Unknown' 时才添加
            # (这可以防止 'wavelength' 或其他非 ROI 列被错误添加)
            if stress_name != 'Unknown':
                all_results.append(result_row)
            
        print(f"  完成 {len(ref_df.columns)} 个 ROIs。")

    # 7. 循环结束后，保存所有结果
    print(f"\n--- 批量处理完成 ---")
    if all_results:
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"✅ 成功保存 {len(output_df)} 行结果到: {OUTPUT_CSV_PATH}")
    else:
        print("未生成任何结果。")

# ----------------------------------------------------------------------
# --- 5. 运行 ---
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()