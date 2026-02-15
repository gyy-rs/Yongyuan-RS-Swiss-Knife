import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def setup_plotting_style(dpi=300):
    """设置统一且美观的科研绘图风格"""
    sns.set_style("whitegrid")
    try:
        plt.rcParams['font.family'] = 'Arial'
    except:
        print("Arial font not found, using default.")
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['axes.unicode_minus'] = False

def get_color_palette():
    """定义一个统一的配色方案"""
    return {
        'primary': '#005f73',
        'secondary': '#0a9396',
        'accent': '#ee9b00',
        'line': '#ae2012',
        'text': '#333333'
    }

def plot_actual_vs_predicted(y_true, y_pred, palette, r2, rmse, output_path='predicted_vs_actual_log_transformed.png'):
    """绘制 真实值 vs. 预测值 散点图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sns.scatterplot(x=y_true, y=y_pred, color=palette['primary'], alpha=0.5, ax=ax, s=40, edgecolor='w')
    
    min_val = min(y_true.min(), y_pred.min()) * 0.9
    max_val = max(y_true.max(), y_pred.max()) * 1.1
    lims = [min_val, max_val]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.plot(lims, lims, color=palette['line'], linestyle='--', lw=2, label='1:1 Line')
    
    text_str = f'$R^2 = {r2:.3f}$\nRMSE = {rmse:.3f}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
            
    ax.set_xlabel('Actual SIF')
    ax.set_ylabel('Predicted SIF')
    ax.set_title('Model Performance on Test Set (with Log Transform)')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"模型性能图已保存至: {output_path}")
    plt.show()

def main():
    """主执行函数"""
    setup_plotting_style(dpi=300)
    palette = get_color_palette()
    
    print("正在加载数据...")
    filepath = '/pg_disk/@open_data/@Paper4.SIF_downscaling/30m/tropomi_footprints_with_zonal_stats_manual_V2.gpkg'
    df = gpd.read_file(filepath)
    
    # 筛选纯净的
    df = df[df["proportion"] > 0.60]

    target_column = 'sif743'
    feature_columns = [
        'Red_L8', 'NIR_L8', 'TIRS1_L8', 'TIRS2_L8', 'Red_S2', 'NIR_S2', 
        'SWIR1_S2', 'SWIR2_S2', 'NDVI_L8', 'NDVI_S2', 'EVI2_L8', 'EVI2_S2', 
        'NIRv_L8', 'NIRv_S2',
        'MSAVI_L8', 'MSAVI_S2', 'OSAVI_L8', 'OSAVI_S2', 'RDVI_L8', 'RDVI_S2',  
        'NDMI_S2', 'GVMI_S2', 'MNDVI_S2', 'SLAVI_S2', 'MSI_S2', 'NMDI_S2'
    ]
    
    # --- 【核心修改】数据清洗 ---
    print(f"原始数据行数: {len(df)}")
    
    # 1. 清理特征和目标中的NaN或无穷大值
    clean_cols = feature_columns + [target_column]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=clean_cols, inplace=True)
    
    # 2. 移除目标值为负数的行，这是解决log1p问题的关键
    df = df[df[target_column] >= 0].copy()
    
    print(f"数据清洗后剩余行数: {len(df)}")

    X = df[feature_columns]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"数据已分割: {len(X_train)}个训练样本, {len(X_test)}个测试样本。")

    print("\n正在对目标变量y进行log(1+y)变换...")
    y_train_log = np.log1p(y_train)

    print("正在配置并训练调优后的随机森林模型 (在log空间)...")
    
    tuned_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    tuned_model.fit(X_train, y_train_log)
    print("模型训练完成。")

    print("\n正在测试集上评估最终模型...")
    y_pred_log = tuned_model.predict(X_test)
    y_pred_original = np.expm1(y_pred_log)
    
    final_r2 = r2_score(y_test, y_pred_original)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_original))
    
    print(f"测试集最终性能 (在原始尺度上评估):")
    print(f"  - R²: {final_r2:.3f}")
    print(f"  - RMSE: {final_rmse:.3f}")

    plot_actual_vs_predicted(y_test, y_pred_original, palette, final_r2, final_rmse)
    
    print("\n全部流程执行完毕。")

if __name__ == '__main__':
    main()