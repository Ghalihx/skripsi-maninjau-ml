// ============================================================================
// SCRIPT GOOGLE EARTH ENGINE (GEE) — SKRIPSI UNIVERSITAS DJUANDA (2026)
// Judul: Komparasi Algoritma Random Forest, XGBoost, dan Support Vector Regression
//        untuk Prediksi Anomali Suhu Permukaan Danau Maninjau Berbasis
//        Google Earth Engine sebagai Indikator Termal Potensi Upwelling
// ============================================================================
// Pembaruan Teknis (Revisi Audit Kode):
// 1. Konstruksi eksplisit ee.FeatureCollection() pada hasil .map()
// 2. Penambahan filter kualitas QC_Day & QC_Night pada MODIS (MOD11A1 & MYD11A1)
// 3. Penambahan filter ketidakpastian termal ST_QA (< 3 K) & masker zona tengah air
//    pada Landsat 8/9 untuk mencegah kontaminasi piksel daratan/pesisir
// 4. Penggunaan scale resolusi asli: ERA5-Land (11132 m), CHIRPS (5566 m),
//    MODIS (1000 m), dan Landsat (30 m)
// ============================================================================

// 1. DEFINISI AREA STUDI (ZONA PELAGIS TENGAH DANAU MANINJAU)
var centerManinjau = ee.Geometry.Point([100.187, -0.315]);
// Buffer 2.0 km di tengah danau (menghindari tepi kaldera/daratan)
var roiCoreWater = centerManinjau.buffer(2000);

// Masker perairan permanen (occurrence > 90%) dari JRC Global Surface Water
var jrcWaterMask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence')
  .gt(90);

Map.centerObject(centerManinjau, 12);
Map.addLayer(roiCoreWater, {color: 'cyan'}, 'ROI Inti Perairan Danau Maninjau (r=2km)');

var startDate = '2016-01-01';
var endDate = '2026-01-01'; // Sampai 31 Desember 2025

// ============================================================================
// 2. FUNGSI KONTROL KUALITAS & EKSTRAKSI LANDSAT 8/9 LEVEL-2 (30 METER)
// ============================================================================
function extractLandsatQualityLST(img) {
  var qa = img.select('QA_PIXEL');
  // Bit 0: Fill, Bit 1: Dilated Cloud, Bit 2: Cirrus, Bit 3: Cloud, Bit 4: Cloud Shadow
  var cloudFree = qa.bitwiseAnd(parseInt('11111', 2)).eq(0);
  // Bit 7: Water confidence / JRC permanent water mask
  
  // Kontrol kualitas termal ST_QA (Skala 0.01 Kelvin -> ambil yang ketidakpastiannya < 3.0 K)
  var stQaKelvin = img.select('ST_QA').multiply(0.01);
  var goodThermalQa = stQaKelvin.lt(3.0);
  
  var validMask = cloudFree.and(goodThermalQa).and(jrcWaterMask);
  
  // Konversi ST_B10 ke Celsius
  var lstCelsius = img.select('ST_B10')
    .multiply(0.00341802)
    .add(149.0)
    .subtract(273.15)
    .rename('landsat_lst_c')
    .updateMask(validMask);
    
  var stats = lstCelsius.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.count(), sharedInputs: true}),
    geometry: roiCoreWater,
    scale: 30,
    maxPixels: 1e9
  });
  
  var qaStats = stQaKelvin.updateMask(validMask).reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roiCoreWater,
    scale: 30,
    maxPixels: 1e9
  });
  
  return ee.Feature(null, {
    'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
    'timestamp': img.get('system:time_start'),
    'spacecraft': img.get('SPACECRAFT_ID'),
    'cloud_cover_scene': img.get('CLOUD_COVER'),
    'landsat_lst_mean_c': stats.get('landsat_lst_c_mean'),
    'landsat_lst_std_c': stats.get('landsat_lst_c_stdDev'),
    'valid_water_pixels': stats.get('landsat_lst_c_count'),
    'mean_st_qa_kelvin': qaStats.get('ST_QA')
  });
}

var l8Col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roiCoreWater)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'));

var l9Col = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterBounds(roiCoreWater)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'));

var landsatMerged = l8Col.merge(l9Col).sort('system:time_start');

// Konstruksi eksplisit sebagai ee.FeatureCollection
var landsatFeatureCol = ee.FeatureCollection(landsatMerged.map(extractLandsatQualityLST))
  .filter(ee.Filter.notNull(['landsat_lst_mean_c']))
  .filter(ee.Filter.gte('valid_water_pixels', 100)); // Minimal 100 piksel air murni bebas awan

print('1. Total Lintasan Citra Landsat 8/9 (2016-2025):', landsatMerged.size());
print('2. Jumlah Observasi Landsat Lolos QC (Valid >= 100 piksel air):', landsatFeatureCol.size());
print('Contoh 5 Observasi Landsat Lolos QC:', landsatFeatureCol.limit(5));

// ============================================================================
// 3. FUNGSI KONTROL KUALITAS MODIS LST (QC_Day & QC_Night)
// ============================================================================
// Bit 0-1: Mandatory QA flags (00 = LST produced, good quality; 01 = other quality)
// Bit 6-7: Average LST error (00 = <= 1K, 01 = <= 2K, 10 = <= 3K)
function maskModisLST(img, lstBandName, qcBandName) {
  var qc = img.select(qcBandName);
  var mandatoryQa = qc.bitwiseAnd(3); // Bit 0-1
  var lstError = qc.rightShift(6).bitwiseAnd(3); // Bit 6-7
  
  // Hanya terima piksel dengan Mandatory QA <= 1 dan LST Error <= 1 (<= 2 Kelvin)
  var validModisMask = mandatoryQa.lte(1).and(lstError.lte(1)).and(jrcWaterMask);
  
  return img.select(lstBandName)
    .multiply(0.02)
    .subtract(273.15)
    .updateMask(validModisMask);
}

// ============================================================================
// 4. EKSTRAKSI DATASET HARIAN TERPADU (ERA5-LAND + CHIRPS + MODIS QC)
// ============================================================================
var era5Daily = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterBounds(roiCoreWater)
  .filterDate(startDate, endDate);

var chirpsDaily = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterBounds(roiCoreWater)
  .filterDate(startDate, endDate);

var modisTerra = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterBounds(roiCoreWater)
  .filterDate(startDate, endDate);

var modisAqua = ee.ImageCollection('MODIS/061/MYD11A1')
  .filterBounds(roiCoreWater)
  .filterDate(startDate, endDate);

var dailyFeatureCol = ee.FeatureCollection(era5Daily.map(function(eraImg) {
  var date = eraImg.date();
  var dateStr = date.format('YYYY-MM-dd');
  var nextDay = date.advance(1, 'day');
  
  // A. Variabel Meteorologi & Reanalisis Termal ERA5-Land (Resolusi asli ~11.1 km = 11132 m)
  var tAirMean = eraImg.select('temperature_2m').subtract(273.15);
  var tAirMin = eraImg.select('temperature_2m_min').subtract(273.15);
  var tAirMax = eraImg.select('temperature_2m_max').subtract(273.15);
  var tLakeMix = eraImg.select('lake_mix_layer_temperature').subtract(273.15);
  var tSkin = eraImg.select('skin_temperature').subtract(273.15);
  
  var u10 = eraImg.select('u_component_of_wind_10m');
  var v10 = eraImg.select('v_component_of_wind_10m');
  var windSpeed = u10.hypot(v10).rename('wind_speed_10m');
  var windDir = u10.atan2(v10).multiply(180.0 / Math.PI).rename('wind_dir_deg');
  
  var solarRadMJ = eraImg.select('surface_net_solar_radiation_sum').divide(1e6);
  var thermalRadMJ = eraImg.select('surface_net_thermal_radiation_sum').divide(1e6);
  var surfacePresHPa = eraImg.select('surface_pressure').divide(100.0);
  var precipEra5Mm = eraImg.select('total_precipitation_sum').multiply(1000.0);
  
  var eraCombined = ee.Image.cat([
    tAirMean.rename('air_temp_mean_c'),
    tAirMin.rename('air_temp_min_c'),
    tAirMax.rename('air_temp_max_c'),
    tLakeMix.rename('era5_lake_mix_temp_c'),
    tSkin.rename('era5_skin_temp_c'),
    u10.rename('u_wind_10m_ms'),
    v10.rename('v_wind_10m_ms'),
    windSpeed,
    windDir,
    solarRadMJ.rename('solar_net_rad_mjm2'),
    thermalRadMJ.rename('thermal_net_rad_mjm2'),
    surfacePresHPa.rename('surface_pressure_hpa'),
    precipEra5Mm.rename('precip_era5_mm')
  ]);
  
  var eraStats = eraCombined.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roiCoreWater,
    scale: 11132, // Resolusi asli ERA5-Land (~0.1 derajat / 11.1 km)
    maxPixels: 1e9
  });
  
  // B. Curah Hujan CHIRPS Daily (Resolusi asli ~0.05 derajat / 5566 m)
  var chirpsImg = chirpsDaily.filterDate(date, nextDay).first();
  var precipChirps = ee.Algorithms.If(
    chirpsImg,
    chirpsImg.select('precipitation').reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roiCoreWater,
      scale: 5566
    }).get('precipitation'),
    null
  );
  
  // C. Suhu Permukaan MODIS Terra (Siang & Malam) dengan Filter QC
  var terraImg = modisTerra.filterDate(date, nextDay).first();
  var terraDayC = ee.Algorithms.If(
    terraImg,
    maskModisLST(terraImg, 'LST_Day_1km', 'QC_Day')
      .reduceRegion({reducer: ee.Reducer.mean(), geometry: roiCoreWater, scale: 1000}).get('LST_Day_1km'),
    null
  );
  var terraNightC = ee.Algorithms.If(
    terraImg,
    maskModisLST(terraImg, 'LST_Night_1km', 'QC_Night')
      .reduceRegion({reducer: ee.Reducer.mean(), geometry: roiCoreWater, scale: 1000}).get('LST_Night_1km'),
    null
  );
  
  // D. Suhu Permukaan MODIS Aqua (Siang & Malam) dengan Filter QC
  var aquaImg = modisAqua.filterDate(date, nextDay).first();
  var aquaDayC = ee.Algorithms.If(
    aquaImg,
    maskModisLST(aquaImg, 'LST_Day_1km', 'QC_Day')
      .reduceRegion({reducer: ee.Reducer.mean(), geometry: roiCoreWater, scale: 1000}).get('LST_Day_1km'),
    null
  );
  var aquaNightC = ee.Algorithms.If(
    aquaImg,
    maskModisLST(aquaImg, 'LST_Night_1km', 'QC_Night')
      .reduceRegion({reducer: ee.Reducer.mean(), geometry: roiCoreWater, scale: 1000}).get('LST_Night_1km'),
    null
  );
  
  return ee.Feature(null, {
    'date': dateStr,
    'year': date.get('year'),
    'month': date.get('month'),
    'day_of_year': date.getRelative('day', 'year').add(1),
    'air_temp_mean_c': eraStats.get('air_temp_mean_c'),
    'air_temp_min_c': eraStats.get('air_temp_min_c'),
    'air_temp_max_c': eraStats.get('air_temp_max_c'),
    'u_wind_10m_ms': eraStats.get('u_wind_10m_ms'),
    'v_wind_10m_ms': eraStats.get('v_wind_10m_ms'),
    'wind_speed_10m': eraStats.get('wind_speed_10m'),
    'wind_dir_deg': eraStats.get('wind_dir_deg'),
    'solar_net_rad_mjm2': eraStats.get('solar_net_rad_mjm2'),
    'thermal_net_rad_mjm2': eraStats.get('thermal_net_rad_mjm2'),
    'surface_pressure_hpa': eraStats.get('surface_pressure_hpa'),
    'precip_era5_mm': eraStats.get('precip_era5_mm'),
    'precip_chirps_mm': precipChirps,
    'era5_lake_mix_temp_c': eraStats.get('era5_lake_mix_temp_c'),
    'era5_skin_temp_c': eraStats.get('era5_skin_temp_c'),
    'modis_terra_day_c': terraDayC,
    'modis_terra_night_c': terraNightC,
    'modis_aqua_day_c': aquaDayC,
    'modis_aqua_night_c': aquaNightC
  });
}));

// Hitung ketersediaan observasi MODIS yang lolos QC
var validTerraDay = dailyFeatureCol.filter(ee.Filter.notNull(['modis_terra_day_c']));
var validTerraNight = dailyFeatureCol.filter(ee.Filter.notNull(['modis_terra_night_c']));
var validAquaDay = dailyFeatureCol.filter(ee.Filter.notNull(['modis_aqua_day_c']));
var validAquaNight = dailyFeatureCol.filter(ee.Filter.notNull(['modis_aqua_night_c']));

print('3. Total Baris Harian Dataset (2016-2025):', dailyFeatureCol.size());
print('4. Jumlah Hari MODIS Lolos QC - Terra Siang:', validTerraDay.size(), '| Terra Malam:', validTerraNight.size());
print('5. Jumlah Hari MODIS Lolos QC - Aqua Siang:', validAquaDay.size(), '| Aqua Malam:', validAquaNight.size());

// ============================================================================
// 5. EKSPOR KE GOOGLE DRIVE (CSV)
// ============================================================================
Export.table.toDrive({
  collection: dailyFeatureCol,
  description: 'Export_Maninjau_Daily_Dataset_2016_2025',
  folder: 'Skripsi_Maninjau_GEE',
  fileNamePrefix: 'maninjau_daily_gee_2016_2025',
  fileFormat: 'CSV',
  selectors: [
    'date', 'year', 'month', 'day_of_year',
    'air_temp_mean_c', 'air_temp_min_c', 'air_temp_max_c',
    'u_wind_10m_ms', 'v_wind_10m_ms', 'wind_speed_10m', 'wind_dir_deg',
    'solar_net_rad_mjm2', 'thermal_net_rad_mjm2', 'surface_pressure_hpa',
    'precip_era5_mm', 'precip_chirps_mm',
    'era5_lake_mix_temp_c', 'era5_skin_temp_c',
    'modis_terra_day_c', 'modis_terra_night_c',
    'modis_aqua_day_c', 'modis_aqua_night_c'
  ]
});

Export.table.toDrive({
  collection: landsatFeatureCol,
  description: 'Export_Maninjau_Landsat_LST_2016_2025',
  folder: 'Skripsi_Maninjau_GEE',
  fileNamePrefix: 'maninjau_landsat_lst_2016_2025',
  fileFormat: 'CSV',
  selectors: [
    'date', 'spacecraft', 'cloud_cover_scene',
    'landsat_lst_mean_c', 'landsat_lst_std_c',
    'valid_water_pixels', 'mean_st_qa_kelvin'
  ]
});
