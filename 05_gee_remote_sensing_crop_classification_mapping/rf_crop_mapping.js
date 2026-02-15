var roi = ee.FeatureCollection("users/xuxq55/guanzhong"),
    table = ee.FeatureCollection("users/xuxq55/label"),
    table2 = ee.FeatureCollection("users/xuxq55/polygon"),
    imageVisParam = {"opacity":1,"bands":["constant","VH_1","VH"],"min":-25,"max":5,"gamma":1},
    imageVisParam2 = {"opacity":1,"bands":["constant","VH_1","VH"],"min":-25,"max":5,"gamma":1},
    imageVisParam3 = {"opacity":1,"bands":["constant","VH_1","VH"],"min":-25,"max":5,"gamma":1};  
  
  // ROI
  roi = roi.union();
  Map.addLayer(roi,{},'shp',false);
  
  
  // Sentinel-1 
  var imgVV = ee.ImageCollection('COPERNICUS/S1_GRD')
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filterBounds(roi)
          .map(function(image) {
            var edge = image.lt(-30.0);
            var maskedImage = image.mask().and(edge.not());
            return image.updateMask(maskedImage);
          });
  
  var asc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
  
  var VV_img = asc.filterDate('2020-10-20', '2021-06-01').select("VV").median().clip(roi);
  var VH_img = asc.filterDate('2020-10-20', '2021-06-01').select("VH").median().clip(roi);
  
  Map.addLayer(VV_img, {min: -25, max: 5}, 'VV_img',false);
  Map.addLayer(VH_img, {min: -25, max: 5}, 'VH_img',false);
  
  
  // S2 Image Cloud Removal and Synthesis
  function maskS2clouds(image) {
    var qa = image.select('QA60');
  
    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;
  
    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
        
    return image.updateMask(mask).divide(10000);
  }
  
  
  // Calculate Features
  // NDVI
  function NDVI(img){
  var ndvi=img.expression(
    "(NIR-R)/(NIR+R)",
    {"R": img.select("B4"),
     "NIR": img.select("B8"),
    }); 
    return img.addBands(ndvi.rename("NDVI"));
  }
  
  // NDWI
  function NDWI(img){
  var ndwi=img.expression(
    "(G-MIR)/(G+MIR)",
    {"G": img.select("B3"),
     "MIR": img.select("B8"),
    });
    return img.addBands(ndwi.rename("NDWI"));
  }
  
  // NDBI
  function NDBI(img){
  var ndbi=img.expression(
    "(SWIR-NIR)/(SWIR-NIR)",
    {"NIR": img.select("B8"),
    "SWIR": img.select("B12"),
    }); 
    return img.addBands(ndbi.rename("NDBI"));
  }
  
  
  //SAVI
  function SAVI(image) {
      var savi = image.expression('(NIR - RED) * (1 + 0.5)/(NIR + RED + 0.5)', {
      'NIR': image.select('B8'),
      'RED': image.select('B4')
      }).float();
      return image.addBands(savi.rename('SAVI'));
  }
  
  //IBI 
  function IBI(image) {
    // Add Index-Based Built-Up Index (IBI)
    var ibiA = image.expression('2 * SWIR1 / (SWIR1 + NIR)', {
      'SWIR1': image.select('B6'),
      'NIR'  : image.select('B5')
    }).rename(['IBI_A']);
   
    var ibiB = image.expression('(NIR / (NIR + RED)) + (GREEN / (GREEN + SWIR1))', {
      'NIR'  : image.select('B8'),
      'RED'  : image.select('B4'),
      'GREEN': image.select('B3'),
      'SWIR1': image.select('B11')
    }).rename(['IBI_B']);
   
    var ibiAB = ibiA.addBands(ibiB);
    var ibi = ibiAB.normalizedDifference(['IBI_A', 'IBI_B']);
    return image.addBands(ibi.rename(['IBI']));
  }
  
  
  //RVI
  function RVI(image){
      var rvi = image.expression('NIR/Red', {
      'NIR': image.select('B8'),
      'Red': image.select('B4')
      });
      return image.addBands(rvi.rename('RVI'));
  }

  //DVI
  function DVI(image){
      var dvi = image.expression('NIR - Red', {
      'NIR': image.select('B8'),
      'Red': image.select('B4')
    }).float();
    return image.addBands(dvi.rename('DVI'));
  }
  
  
  
  var dataset = ee.ImageCollection('COPERNICUS/S2_SR')
                    .filterDate('2021-03-20', '2021-06-01')
                    .filterBounds(roi)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
                    .map(maskS2clouds)
                    .map(NDVI)
                    .map(NDWI)
                    .map(NDBI)
                    .map(SAVI)
                    .map(IBI)
                    .map(RVI)
                    .map(DVI);
                    
                    
  dataset = dataset.median().clip(roi);
  // print(dataset);
  
  var visualization = {
    min: 0.0,
    max: 0.3,
    bands: ['B4', 'B3', 'B2'],
  };
  
  Map.addLayer(dataset,visualization,'S2',false);
  
  
  
  //   Terrain Features, DEM, Slope

  var NASADEM = ee.Image('NASA/NASADEM_HGT/001');
  var DEM = NASADEM.select('elevation');
  var slope = ee.Terrain.slope(DEM);
  
  
  // Texture Features
  
  var B8 = dataset.select('B8').multiply(100).toInt16();
  var glcm = B8.glcmTexture({size:3});
  // print(glcm);
  
  var contrast = glcm.select('B8_contrast');
  var var_ = glcm.select('B8_var');
  var savg = glcm.select('B8_savg');
  var dvar = glcm.select('B8_dvar');
  
  var palette = ['040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
                '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
                '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
                'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
                'ff0000', 'de0101', 'c21301', 'a71001', '911003'];
    
    
  Map.addLayer(contrast,{min:0, max:100, palette:palette},'contrast',false);
  Map.addLayer(savg,{min:0, max:100, palette:palette},'savg',false);
  Map.addLayer(var_ ,{min:0, max:100, palette:palette},'var_',false);
  Map.addLayer(dvar ,{min:0, max:100, palette:palette},'dvar',false);
  
  
  
  // Select Bandwidth
  var image = dataset.addBands(VV_img.rename("VV"))
  .addBands(VH_img.rename("VH"))
  .addBands(DEM.rename("DEM"))
  .addBands(slope.rename("slope"))
  .addBands(contrast.rename("contrast"))
  .addBands(var_.rename("var"))
  .addBands(savg.rename("savg"))
  .addBands(dvar.rename("dvar"))
  .clip(roi);
  
  
  var bands = ['B2','B3','B4','B8','B11','B12','NDVI','NDWI','NDBI','SAVI','IBI','RVI','DVI','VV','VH','DEM','slope','contrast','var','savg','dvar'];
  var img = image.select(bands);
  // print(img);
  
  
  //The computational load is too high. It is recommended to export the features (img) to an asset and then import them for classification. When performing classification, comment out the previous feature calculations.

  // Export.image.toAsset({
  // image: img,
  // description: 'img',
  // assetId: 'classification',
  // scale: 10,
  // region: roi,
  // });

  
  var training = img.reduceRegions({
    collection:table.merge(table2),
    reducer :ee.Reducer.first(),
    scale:10,
    tileScale :4
  });
  
  
  var trainging2 = training.filter(ee.Filter.notNull(bands));
  // print(trainging2);
  
  
  //  // Add a random attribute to the training feature set with values ranging from 0 to 1

  var withRandom = trainging2.randomColumn({
    columnName:'random',
    seed:0,
    distribution: 'uniform',
  });
  
  
  var split = 0.8; 
  var trainingPartition = withRandom.filter(ee.Filter.lt('random', split));
  var validationPartition = withRandom.filter(ee.Filter.gte('random', split));
  
  // train the RF classification model
  var classifier = ee.Classifier.smileRandomForest(100)
      .train({
        features: trainingPartition, 
        classProperty: 'id', 
        inputProperties: bands
      });
  print(classifier);
  
  // Classify the input imagery.
  var classified = img.classify(classifier);
  print(classified);
  
  validationPartition = classified.reduceRegions({
    collection:validationPartition,
    reducer:ee.Reducer.first(),
    scale:10,
    tileScale:4
  });
  
  // print(validationPartition);
  
  var errorMatrix = validationPartition.errorMatrix('id','first',[0,1,2,3,4,5,6]);
  var overall = errorMatrix.accuracy();
  var producer = errorMatrix.producersAccuracy();
  var user = errorMatrix.consumersAccuracy();
  var kappa = errorMatrix.kappa();
  
  print(overall,'OA');
  print(producer,'producer accuracy');
  print(user,'user accuracy');
  print(kappa,'Kappa');
  
  Export.image.toDrive({
    image: classified.toInt(),
    description:"classified_s2_epsg_4326",
    folder: 'classified',
    fileNamePrefix: "s2_epsg_4326",
    region:roi.geometry().bounds(),
    scale :20,
    maxPixels:1e13
  });
  
  // Define a palette for the IGBP classification.
  var igbpPalette = [
    '#D4F2E7', // 0 
    '#E6E6FA', // 1
    'green',//2
    '#7B68EE',//3
    '#00FF00', 
    'blue',//5
    '#DC143C'//6 
    
  ];
  // // 1-rice, 2-corn, 3-lily, 4-others
  Map.addLayer(classified, {palette: igbpPalette, min: 0, max:6}, 'classification');
