/**
 * Indigirka Daily Comprehensive Export (2018-2025) - V3
 * * @author yongyuan gao
 * * Updates:
 * 1. ERA5: Calculate gee_par (W/m²), logic is SSRD(W/m²) * 0.5.
 * 2. MODIS: All reflectance bands are pre-multiplied by 0.0001. 
 * The exported NIRv/NDVI are actual decimals (0-1), no subsequent division needed.
 */

// 1. Define spatial region (ROI)
var roi = ee.Geometry.Polygon([[
  [140.0, 72.0], [150.0, 72.0], [150.0, 71.0], 
  [147.5, 71.0], [147.5, 70.0], [140.0, 70.0], [140.0, 72.0]  
]], null, false);

// 2. Time range configuration
var startDate = ee.Date('2018-01-01');
var endDate = ee.Date('2025-12-31'); 
var dayList = ee.List.sequence(0, endDate.difference(startDate, 'day').subtract(1));

// 3. Data source loading and function definitions

// --- ERA5 Land processing (Calculate gee_par) ---
var processERA5 = function(img) {
  // Basic meteorology
  var t2mC = img.select('temperature_2m').subtract(273.15);
  var d2mC = img.select('dewpoint_temperature_2m').subtract(273.15);
  var spKPa = img.select('surface_pressure').divide(1000); 
  
  // Calculate VPD
  var e_s = t2mC.expression('0.6108 * exp(17.27 * T / (T + 237.3))', {'T': t2mC});
  var e_a = d2mC.expression('0.6108 * exp(17.27 * Td / (Td + 237.3))', {'Td': d2mC});
  var vpd = e_s.subtract(e_a).rename('VPD').clamp(0, 10);
  
  // --- [Modification 1] Calculate gee_par (W/m^2) ---
  // J/m^2 (daily) / 86400 = W/m^2 (mean)
  // PAR approx 0.5 * Total Shortwave
  var gee_par = img.select('surface_solar_radiation_downwards_sum')
                     .divide(86400)
                     .multiply(0.48) 
                     .rename('gee_par'); 
  
  // Precipitation (mm)
  var precip = img.select('total_precipitation_sum')
                    .multiply(1000)
                    .rename('Precipitation'); 
                    
  // Soil moisture
  var sm1 = img.select('volumetric_soil_water_layer_1').rename('SM_0_7cm');
  var sm2 = img.select('volumetric_soil_water_layer_2').rename('SM_7_28cm');

  return img.addBands([
      vpd, 
      t2mC.rename('temp_2m'), 
      spKPa.rename('pressure_kPa'),
      gee_par, // Output gee_par
      precip,
      sm1,
      sm2
    ])
    .copyProperties(img, ['system:time_start']);
};

// --- MODIS VIs (Pre-scaling) ---
var addVIs = function(img) {
  // [Modification 2] Multiply all bands by 0.0001 first to restore true reflectance (0-1)
  // Calculated NIRv will be actual values (0.0x - 0.x), no need to divide by 10000 after export
  var scaledImg = img.multiply(0.0001);
  
  var b1 = scaledImg.select('Nadir_Reflectance_Band1'); 
  var b2 = scaledImg.select('Nadir_Reflectance_Band2'); 
  var b3 = scaledImg.select('Nadir_Reflectance_Band3'); 
  var b4 = scaledImg.select('Nadir_Reflectance_Band4'); 
  var b6 = scaledImg.select('Nadir_Reflectance_Band6'); 
  
  var ndvi = scaledImg.normalizedDifference(['Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band1']).rename('NDVI');
  // At this point, b2 and ndvi are real decimals, so their product is also a real decimal
  var nirv = ndvi.multiply(b2).rename('NIRv');
  
  var evi = scaledImg.expression('2.5 * ((N-R)/(N + 6*R - 7.5*B + 1))', {'N':b2,'R':b1,'B':b3}).rename('EVI');
  var evi2 = scaledImg.expression('2.5 * (N-R)/(N + 2.4*R + 1)', {'N':b2,'R':b1}).rename('EVI2');
  var lswi = scaledImg.normalizedDifference(['Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band6']).rename('LSWI');
  var msi = b6.divide(b2).rename('MSI');
  var mndwi = scaledImg.normalizedDifference(['Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band6']).rename('MNDWI');
  var ndwi = scaledImg.normalizedDifference(['Nadir_Reflectance_Band4', 'Nadir_Reflectance_Band2']).rename('NDWI');
  var savi = scaledImg.expression('1.5 * (N-R)/(N + R + 0.5)', {'N':b2,'R':b1}).rename('SAVI');
  var csi = scaledImg.expression('2.5 * (N-R)/(N+R) * (B/R)', {'N':b2,'R':b1,'B':b3}).rename('CSI');
  
  return img.addBands([ndvi, nirv, evi, evi2, lswi, msi, mndwi, ndwi, savi, csi])
            .copyProperties(img, ['system:time_start']);
};

// --- Collection loading ---
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
             .filterDate(startDate, endDate)
             .map(processERA5);

var modisVI = ee.ImageCollection("MODIS/061/MCD43A4")
                .filterDate(startDate, endDate)
                .map(addVIs);

var modisLST = ee.ImageCollection("MODIS/061/MOD11A1")
  .filterDate(startDate, endDate)
  .select(['LST_Day_1km'],['LST_Day'])
  .map(function(img){
    return img.multiply(0.02)
              .subtract(273.15)
              .copyProperties(img, ['system:time_start']);
  });

var modisLAI = ee.ImageCollection("MODIS/061/MCD15A3H")
  .filterDate(startDate, endDate)
  .select(['Lai'],['LAI'])
  .map(function(img){
    return img.multiply(0.1)
              .copyProperties(img, ['system:time_start']);
  });

// 4. Daily iteration logic
var dailyData = dayList.map(function(n) {
  var date = startDate.advance(n, 'day');
  var nextDay = date.advance(1, 'day');
  
  // ERA5
  var cli = era5.filterDate(date, nextDay)
                .select(['temp_2m', 'VPD', 'pressure_kPa', 'gee_par', 'Precipitation', 'SM_0_7cm', 'SM_7_28cm'])
                .mean();
  
  // MODIS VIs
  var vi = modisVI.filterDate(date, nextDay)
                  .select(['NDVI','NIRv','EVI','EVI2','LSWI','MSI','MNDWI','NDWI','SAVI','CSI'])
                  .mean();
  
  // LST
  var lst = modisLST.filterDate(date, nextDay).mean();
  
  // LAI
  var lai = modisLAI.filterDate(date.advance(-4, 'day'), date.advance(4, 'day')).mean();
  
  // Combine
  var combined = ee.Image.cat([
    cli, 
    vi, 
    lst.rename('LST_Day'),
    lai.rename('LAI')
  ]);
  
  var stats = combined.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 1000, 
    maxPixels: 1e10,
    tileScale: 4
  });
  
  return ee.Feature(null, stats).set({
    'date': date.format('YYYY-MM-DD'),
    'year': date.get('year'),
    'month': date.get('month'),
    'day': date.get('day')
  });
});

// 5. Export
Export.table.toDrive({
  collection: ee.FeatureCollection(dailyData),
  description: 'Indigirka_Daily_Final_ERA5_PAR_Corrected_NIRv',
  folder: '2026uzh',
  fileFormat: 'CSV',
  selectors: ['date', 'year', 'month', 'day', 
              'temp_2m', 'pressure_kPa', 'VPD', 'gee_par', 'Precipitation', 'SM_0_7cm', 'SM_7_28cm', // PAR and Precip
              'LST_Day', 'LAI', 
              'NDVI', 'NIRv', 'EVI', 'EVI2', 'LSWI', 'MSI', 'MNDWI', 'NDWI', 'SAVI', 'CSI'] 
});

print('Final task submitted: gee_par (W/m2), NIRv (0-1 scaled)');