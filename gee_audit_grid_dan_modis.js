// ============================================================================
// SCRIPT AUDIT SPASIAL GRID ERA5-LAND & KUALITAS PIKSEL MODIS (GEE)
// Tujuan Audit Tambahan:
// 1. Memeriksa secara spasial apakah sel grid ERA5-Land (~0.1 x 0.1 derajat /
//    11.1 km) yang mencakup ROI Danau Maninjau benar-benar merepresentasikan
//    perairan danau (fraksi air JRC per sel grid, elevasi SRTM, parameter FLake:
//    lake_mix_layer_depth, lake_bottom_temperature, lake_mix_layer_temperature).
// 2. Mengekstrak informasi jumlah piksel valid (count), simpangan baku spasial
//    (stdDev), waktu pengamatan lokal (view_time), dan sudut pandang sensor
//    (view_angle) pada MODIS Terra (MOD11A1) & Aqua (MYD11A1) periode 2016-2025.
// ============================================================================

// 1. DEFINISI GEOMETRI DANAU MANINJAU & ROI INTI PERAIRAN
var centerManinjau = ee.Geometry.Point([100.187, -0.315]);
var roiCoreWater = centerManinjau.buffer(2000); // Buffer 2 km di tengah danau
var roiCalderaBox = centerManinjau.buffer(12000).bounds(); // Area 24x24 km sekitar kaldera

var startDate = '2016-01-01';
var endDate = '2026-01-01';

// Masker perairan permanen JRC (occurrence > 90%) pada resolusi 30m
var jrcWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(90);
var jrcWaterFloat = jrcWater.unmask(0).toFloat().rename('water_fraction');

// DEM SRTM 30m untuk memeriksa topografi kaldera di setiap sel grid ERA5-Land
var srtmDem = ee.Image('USGS/SRTMGL1_003').select('elevation');

Map.centerObject(centerManinjau, 12);
Map.addLayer(jrcWater.selfMask(), {palette: ['0055ff']}, '1. Masker Perairan Permanen JRC (>90%)');
Map.addLayer(roiCoreWater, {color: 'cyan'}, '2. ROI Inti Perairan (r = 2 km)');

// ============================================================================
// BAGIAN A: AUDIT SPASIAL SEL GRID ERA5-LAND DI KAWASAN DANAU MANINJAU
// ============================================================================
var era5Col = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterBounds(roiCalderaBox)
  .filterDate(startDate, endDate);

var era5SampleImg = era5Col.first();
var era5Proj = era5SampleImg.select('lake_mix_layer_temperature').projection();

print('=== AUDIT 1: PROYEKSI & RESOLUSI ASLI ERA5-LAND ===');
print('CRS ERA5-Land:', era5Proj.crs());
print('Nominal Scale ERA5-Land (meter):', era5Proj.nominalScale());

// Hitung rata-rata multi-tahun (2016-2025) parameter danau FLake & suhu di ERA5-Land
var era5MultiYearMean = era5Col.select([
  'lake_mix_layer_temperature',
  'lake_bottom_temperature',
  'lake_total_layer_temperature',
  'lake_mix_layer_depth',
  'skin_temperature',
  'temperature_2m'
]).mean();

// Konversi suhu Kelvin ke Celsius
var era5AuditImg = ee.Image.cat([
  era5MultiYearMean.select('lake_mix_layer_temperature').subtract(273.15).rename('mean_lake_mix_temp_c'),
  era5MultiYearMean.select('lake_bottom_temperature').subtract(273.15).rename('mean_lake_bottom_temp_c'),
  era5MultiYearMean.select('lake_total_layer_temperature').subtract(273.15).rename('mean_lake_total_temp_c'),
  era5MultiYearMean.select('lake_mix_layer_depth').rename('mean_lake_mix_depth_m'),
  era5MultiYearMean.select('skin_temperature').subtract(273.15).rename('mean_skin_temp_c'),
  era5MultiYearMean.select('temperature_2m').subtract(273.15).rename('mean_air_temp_2m_c'),
  ee.Image.pixelLonLat().rename(['grid_center_lon', 'grid_center_lat'])
]).reproject(era5Proj);

// Buat poligon vektor dari setiap sel grid ERA5-Land di sekitar kaldera Maninjau
var gridIdImg = ee.Image.pixelLonLat()
  .select('longitude').multiply(100).round().toInt().multiply(100000)
  .add(ee.Image.pixelLonLat().select('latitude').multiply(100).round().toInt())
  .rename('grid_id')
  .reproject(era5Proj);

var era5GridVectors = gridIdImg.reduceToVectors({
  geometry: roiCalderaBox,
  crs: era5Proj,
  scale: era5Proj.nominalScale(),
  geometryType: 'polygon',
  eightConnected: false,
  labelProperty: 'grid_id',
  maxPixels: 1e9
});

// Untuk setiap sel grid ERA5-Land (~11.1 x 11.1 km), hitung:
// 1. Persentase luas perairan Danau Maninjau (JRC > 90%) di dalam sel grid tersebut
// 2. Luas air dalam km2 di dalam sel grid tersebut
// 3. Apakah sel grid beririsan dengan roiCoreWater (r=2km)
// 4. Rata-rata elevasi SRTM (m) di dalam sel grid
// 5. Nilai variabel FLake ERA5-Land pada sel grid tersebut
var era5GridAuditCol = ee.FeatureCollection(era5GridVectors.map(function(cellFeat) {
  var cellGeom = cellFeat.geometry();
  var intersectsRoiCore = cellGeom.intersects(roiCoreWater, 1);
  var roiOverlapAreaKm2 = cellGeom.intersection(roiCoreWater, 1).area(1).divide(1e6);
  var cellAreaKm2 = cellGeom.area(1).divide(1e6);
  
  // Fraksi air JRC pada resolusi 30m di dalam sel grid ERA5-Land
  var waterFrac = jrcWaterFloat.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: cellGeom,
    scale: 30,
    maxPixels: 1e9
  }).get('water_fraction');
  
  // Elevasi SRTM rata-rata & simpangan baku di dalam sel grid
  var elevStats = srtmDem.reduceRegion({
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
    geometry: cellGeom,
    scale: 90,
    maxPixels: 1e9
  });
  
  // Nilai variabel ERA5-Land pada titik pusat sel grid
  var eraVals = era5AuditImg.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: cellGeom.centroid(1),
    crs: era5Proj,
    scale: era5Proj.nominalScale()
  });
  
  return cellFeat.set({
    'intersects_roi_core_2km': intersectsRoiCore,
    'roi_core_overlap_km2': roiOverlapAreaKm2,
    'cell_area_km2': cellAreaKm2,
    'jrc_water_fraction_pct': ee.Number(waterFrac).multiply(100),
    'jrc_water_area_km2': ee.Number(waterFrac).multiply(cellAreaKm2),
    'srtm_elev_mean_m': elevStats.get('elevation_mean'),
    'srtm_elev_std_m': elevStats.get('elevation_stdDev'),
    'grid_center_lon': eraVals.get('grid_center_lon'),
    'grid_center_lat': eraVals.get('grid_center_lat'),
    'mean_lake_mix_temp_c': eraVals.get('mean_lake_mix_temp_c'),
    'mean_lake_bottom_temp_c': eraVals.get('mean_lake_bottom_temp_c'),
    'mean_lake_total_temp_c': eraVals.get('mean_lake_total_temp_c'),
    'mean_lake_mix_depth_m': eraVals.get('mean_lake_mix_depth_m'),
    'mean_skin_temp_c': eraVals.get('mean_skin_temp_c'),
    'mean_air_temp_2m_c': eraVals.get('mean_air_temp_2m_c')
  });
})).sort('jrc_water_fraction_pct', false);

Map.addLayer(era5GridAuditCol, {color: 'yellow'}, '3. Batas Sel Grid ERA5-Land (~11.1 km)');
Map.addLayer(
  era5GridAuditCol.filter(ee.Filter.eq('intersects_roi_core_2km', true)),
  {color: 'red'},
  '4. Sel Grid ERA5-Land yang Mencakup ROI Inti'
);

print('=== AUDIT 2: DAFTAR SEL GRID ERA5-LAND DI SEKITAR DANAU MANINJAU ===');
print('Jumlah Sel Grid ERA5-Land yang Beririsan dengan ROI Inti (r=2km):',
  era5GridAuditCol.filter(ee.Filter.eq('intersects_roi_core_2km', true)).size());
print('Tabel Atribut Sel Grid ERA5-Land (Diurutkan dari Fraksi Air Terbesar):', era5GridAuditCol);

// ============================================================================
// BAGIAN B: AUDIT JUMLAH PIKSEL VALID, STD SPASIAL, & SUDUT PANDANG MODIS
// ============================================================================
var modisTerra = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterBounds(roiCoreWater)
  .filterDate(startDate, endDate);

var modisAqua = ee.ImageCollection('MODIS/061/MYD11A1')
  .filterBounds(roiCoreWater)
  .filterDate(startDate, endDate);

// Visualisasi grid piksel MODIS 1 km di dalam ROI Inti
var modisProj = modisTerra.first().select('LST_Day_1km').projection();
var totalModisWaterPixelsInRoi = jrcWater.updateMask(jrcWater).reduceRegion({
  reducer: ee.Reducer.count(),
  geometry: roiCoreWater,
  crs: modisProj,
  scale: modisProj.nominalScale()
}).get('occurrence');
print('=== AUDIT 3: KAPASITAS MAKSIMUM PIKSEL MODIS 1 KM DI DALAM ROI (r=2km) ===');
print('Total Piksel Air MODIS (~926.6 m) Maksimum di dalam ROI r=2km:', totalModisWaterPixelsInRoi);

function extractModisStats(img, lstBand, qcBand, viewTimeBand, viewAngleBand, prefix) {
  var qc = img.select(qcBand);
  var mandatoryQa = qc.bitwiseAnd(3);
  var lstError = qc.rightShift(6).bitwiseAnd(3);
  var validMask = mandatoryQa.lte(1).and(lstError.lte(1)).and(jrcWater);
  
  var lstC = img.select(lstBand).multiply(0.02).subtract(273.15).updateMask(validMask).rename('lst_c');
  var vTime = img.select(viewTimeBand).multiply(0.1).updateMask(validMask).rename('view_time');
  var vAngle = img.select(viewAngleBand).subtract(65).updateMask(validMask).rename('view_angle');
  
  var combined = ee.Image.cat([lstC, vTime, vAngle]);
  
  var stats = combined.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.count(), sharedInputs: true})
      .combine({reducer2: ee.Reducer.minMax(), sharedInputs: true}),
    geometry: roiCoreWater,
    crs: modisProj,
    scale: modisProj.nominalScale(),
    maxPixels: 1e9
  });
  
  var out = {};
  out[prefix + '_mean_c'] = stats.get('lst_c_mean');
  out[prefix + '_std_c'] = stats.get('lst_c_stdDev');
  out[prefix + '_min_c'] = stats.get('lst_c_min');
  out[prefix + '_max_c'] = stats.get('lst_c_max');
  out[prefix + '_valid_pixels'] = stats.get('lst_c_count');
  out[prefix + '_view_time_hr'] = stats.get('view_time_mean');
  out[prefix + '_view_angle_deg'] = stats.get('view_angle_mean');
  return ee.Dictionary(out);
}

var emptyTerraDay = ee.Dictionary({
  'terra_day_mean_c': null, 'terra_day_std_c': null, 'terra_day_min_c': null, 'terra_day_max_c': null,
  'terra_day_valid_pixels': 0, 'terra_day_view_time_hr': null, 'terra_day_view_angle_deg': null
});
var emptyTerraNight = ee.Dictionary({
  'terra_night_mean_c': null, 'terra_night_std_c': null, 'terra_night_min_c': null, 'terra_night_max_c': null,
  'terra_night_valid_pixels': 0, 'terra_night_view_time_hr': null, 'terra_night_view_angle_deg': null
});
var emptyAquaDay = ee.Dictionary({
  'aqua_day_mean_c': null, 'aqua_day_std_c': null, 'aqua_day_min_c': null, 'aqua_day_max_c': null,
  'aqua_day_valid_pixels': 0, 'aqua_day_view_time_hr': null, 'aqua_day_view_angle_deg': null
});
var emptyAquaNight = ee.Dictionary({
  'aqua_night_mean_c': null, 'aqua_night_std_c': null, 'aqua_night_min_c': null, 'aqua_night_max_c': null,
  'aqua_night_valid_pixels': 0, 'aqua_night_view_time_hr': null, 'aqua_night_view_angle_deg': null
});

var modisAuditDailyCol = ee.FeatureCollection(era5Col.map(function(eraImg) {
  var date = eraImg.date();
  var dateStr = date.format('YYYY-MM-dd');
  var nextDay = date.advance(1, 'day');
  
  var terraImg = modisTerra.filterDate(date, nextDay).first();
  var tDayDict = ee.Dictionary(ee.Algorithms.If(
    terraImg,
    extractModisStats(terraImg, 'LST_Day_1km', 'QC_Day', 'Day_view_time', 'Day_view_angle', 'terra_day'),
    emptyTerraDay
  ));
  var tNightDict = ee.Dictionary(ee.Algorithms.If(
    terraImg,
    extractModisStats(terraImg, 'LST_Night_1km', 'QC_Night', 'Night_view_time', 'Night_view_angle', 'terra_night'),
    emptyTerraNight
  ));
  
  var aquaImg = modisAqua.filterDate(date, nextDay).first();
  var aDayDict = ee.Dictionary(ee.Algorithms.If(
    aquaImg,
    extractModisStats(aquaImg, 'LST_Day_1km', 'QC_Day', 'Day_view_time', 'Day_view_angle', 'aqua_day'),
    emptyAquaDay
  ));
  var aNightDict = ee.Dictionary(ee.Algorithms.If(
    aquaImg,
    extractModisStats(aquaImg, 'LST_Night_1km', 'QC_Night', 'Night_view_time', 'Night_view_angle', 'aqua_night'),
    emptyAquaNight
  ));
  
  var allProps = ee.Dictionary({'date': dateStr})
    .combine(tDayDict)
    .combine(tNightDict)
    .combine(aDayDict)
    .combine(aNightDict);
    
  return ee.Feature(null, allProps);
}));

// ============================================================================
// BAGIAN C: EKSPOR KEDUA TABEL AUDIT KE GOOGLE DRIVE (CSV)
// ============================================================================
Export.table.toDrive({
  collection: era5GridAuditCol,
  description: 'Audit_1_Grid_ERA5_Land_Maninjau',
  folder: 'Skripsi_Maninjau_GEE',
  fileNamePrefix: 'maninjau_audit_grid_era5',
  fileFormat: 'CSV',
  selectors: [
    'grid_id', 'intersects_roi_core_2km', 'roi_core_overlap_km2',
    'cell_area_km2', 'jrc_water_fraction_pct', 'jrc_water_area_km2',
    'srtm_elev_mean_m', 'srtm_elev_std_m',
    'grid_center_lon', 'grid_center_lat',
    'mean_lake_mix_temp_c', 'mean_lake_bottom_temp_c',
    'mean_lake_total_temp_c', 'mean_lake_mix_depth_m',
    'mean_skin_temp_c', 'mean_air_temp_2m_c'
  ]
});

Export.table.toDrive({
  collection: modisAuditDailyCol,
  description: 'Audit_2_MODIS_Pixel_Count_Maninjau_2016_2025',
  folder: 'Skripsi_Maninjau_GEE',
  fileNamePrefix: 'maninjau_audit_modis_pixels_2016_2025',
  fileFormat: 'CSV',
  selectors: [
    'date',
    'terra_day_mean_c', 'terra_day_std_c', 'terra_day_min_c', 'terra_day_max_c',
    'terra_day_valid_pixels', 'terra_day_view_time_hr', 'terra_day_view_angle_deg',
    'terra_night_mean_c', 'terra_night_std_c', 'terra_night_min_c', 'terra_night_max_c',
    'terra_night_valid_pixels', 'terra_night_view_time_hr', 'terra_night_view_angle_deg',
    'aqua_day_mean_c', 'aqua_day_std_c', 'aqua_day_min_c', 'aqua_day_max_c',
    'aqua_day_valid_pixels', 'aqua_day_view_time_hr', 'aqua_day_view_angle_deg',
    'aqua_night_mean_c', 'aqua_night_std_c', 'aqua_night_min_c', 'aqua_night_max_c',
    'aqua_night_valid_pixels', 'aqua_night_view_time_hr', 'aqua_night_view_angle_deg'
  ]
});
