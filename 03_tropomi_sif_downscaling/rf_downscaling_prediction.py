import os
import glob
import joblib
import rasterio
import numpy as np
import warnings

# 忽略 rasterio 在处理某些元数据时可能产生的警告
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def batch_predict_geotiffs(model_path, input_dir, output_dir, feature_columns):
    """
    使用已保存的模型对指定目录下的所有GeoTIFF文件进行批量预测。

    Args:
        model_path (str): 已保存的scikit-learn模型文件路径 (.joblib)。
        input_dir (str): 包含输入GeoTIFF文件的目录。
        output_dir (str): 用于保存预测结果的目录。
        feature_columns (list): 模型训练时使用的特征名称列表，用于验证波段数。
    """
    # --- 1. 加载模型 ---
    print(f"正在从 {model_path} 加载模型...")
    try:
        model = joblib.load(model_path)
        print("模型加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}。请检查路径是否正确。")
        return
    
    # --- 2. 准备输入和输出目录 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 查找所有tif文件
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
    if not tif_files:
        print(f"警告：在目录 {input_dir} 中未找到任何 .tif 文件。")
        return

    print(f"\n找到 {len(tif_files)} 个 .tif 文件，开始批量预测...")
    num_features = len(feature_columns)

    # --- 3. 循环处理每个TIF文件 ---
    for i, tif_path in enumerate(tif_files):
        print(f"\n--- [{i+1}/{len(tif_files)}] 正在处理: {os.path.basename(tif_path)} ---")
        
        try:
            with rasterio.open(tif_path) as src:
                # --- 3a. 验证波段数 ---
                if src.count != num_features:
                    print(f"  -> 警告: 文件波段数 ({src.count}) 与模型所需特征数 ({num_features}) 不匹配。跳过此文件。")
                    # 关键假设：我们强假定TIF的波段顺序与 feature_columns 列表的顺序完全一致。
                    continue

                # --- 3b. 读取数据和元数据 ---
                profile = src.profile
                nodata_value = src.nodata
                
                # 读取所有波段数据
                data = src.read() # 形状: (bands, height, width)

                # 如果文件中没有定义NoData值，我们选择一个（例如-9999）
                if nodata_value is None:
                    nodata_value = -9999.0
                    print(f"  -> 警告: 文件未定义NoData值，将使用 {nodata_value} 作为NoData值。")
                
                # 创建一个有效数据的掩码 (mask)
                # 假设所有波段的NoData位置都一样，用第一个波段来创建掩码
                mask = data[0] != nodata_value

                # --- 3c. 数据重塑以适配模型输入 ---
                # 将 (bands, height, width) -> (height, width, bands)
                transposed_data = np.transpose(data, (1, 2, 0))
                
                # 将 (height, width, bands) -> (n_pixels, n_features)
                # 只选择有效像素进行重塑
                reshaped_data = transposed_data[mask]

                if reshaped_data.shape[0] == 0:
                    print("  -> 警告: 文件中没有有效的像素数据。跳过此文件。")
                    continue
                
                print(f"  -> 找到 {reshaped_data.shape[0]} 个有效像素进行预测。")

                # --- 3d. 批量预测 ---
                predictions = model.predict(reshaped_data)

                # --- 3e. 结果重构与保存 ---
                # 创建一个与输入栅格相同大小的输出数组，并用NoData值填充
                output_image = np.full((src.height, src.width), nodata_value, dtype=np.float32)

                # 将预测结果填充回有效像素的位置
                output_image[mask] = predictions

                # 更新输出文件的元数据
                profile.update(
                    dtype=rasterio.float32,
                    count=1, # 输出是单波段影像
                    compress='lzw', # 使用无损压缩
                    nodata=nodata_value
                )
                
                # 定义输出文件路径
                output_filename = os.path.basename(tif_path).replace('.tif', '_predicted_SIF.tif')
                output_path = os.path.join(output_dir, output_filename)
                
                # 保存结果
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(output_image, 1)

                print(f"  -> 预测完成，结果已保存至: {output_path}")

        except Exception as e:
            print(f"  -> 处理文件 {tif_path} 时发生错误: {e}")

    print("\n--- 所有文件处理完毕！ ---")


if __name__ == '__main__':
    # --- 用户配置区 ---

    # 1. 定义模型和数据路径
    MODEL_PATH = 'random_forest_sif_model.joblib'
    INPUT_DIRECTORY = '/pg_disk/@open_data/@Paper4.SIF_downscaling/30m/SGed_2022_Mean_L30_S30_VIs'
    OUTPUT_DIRECTORY = '/pg_disk/@open_data/@Paper4.SIF_downscaling/30m/Pred_SIF/predicted_sif_results_2022' # 结果将保存在当前目录下的这个文件夹里

    # 2. 定义模型训练时使用的特征列（必须与模型训练时完全一致）
    # ！！！至关重要的假设：TIF文件中的波段顺序必须与此列表顺序完全一致！！！
    FEATURE_COLUMNS = [
        'Red_L8', 'NIR_L8', 'TIRS1_L8', 'TIRS2_L8', 'Red_S2', 'NIR_S2', 
        'SWIR1_S2', 'SWIR2_S2', 'NDVI_L8', 'NDVI_S2', 'EVI2_L8', 'EVI2_S2', 
        'NIRv_L8', 'NIRv_S2', 'DVI_L8', 'DVI_S2', 'SAVI_L8', 'SAVI_S2', 
        'MSAVI_L8', 'MSAVI_S2', 'OSAVI_L8', 'OSAVI_S2', 'RDVI_L8', 'RDVI_S2', 
        'SR_L8', 'SR_S2', 'IPVI_L8', 'IPVI_S2', 'NLI_L8', 'NLI_S2', 
        'TVI_L8', 'TVI_S2', 'WDRVI_L8', 'WDRVI_S2', 'AFVI1600_S2', 'AFVI2100_S2', 
        'NDMI_S2', 'GVMI_S2', 'MNDVI_S2', 'SLAVI_S2', 'MSI_S2', 'NMDI_S2'
    ]
    
    # --- 执行批量预测 ---
    # batch_predict_geotiffs(MODEL_PATH, INPUT_DIRECTORY, OUTPUT_DIRECTORY, FEATURE_COLUMNS)


    INPUT_DIRECTORY = '/pg_disk/@open_data/@Paper4.SIF_downscaling/30m/SGed_5yr_Mean_L30_S30_VIs'
    OUTPUT_DIRECTORY = '/pg_disk/@open_data/@Paper4.SIF_downscaling/30m/Pred_SIF/predicted_sif_results_5yr' # 结果将保存在当前目录下的这个文件夹里
    batch_predict_geotiffs(MODEL_PATH, INPUT_DIRECTORY, OUTPUT_DIRECTORY, FEATURE_COLUMNS)
