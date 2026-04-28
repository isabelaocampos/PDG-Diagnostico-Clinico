// File: mirth/transformaciones/construir_json.js
// Script de Mirth Connect para construir el JSON que será enviado a FastAPI.
try {
    logger.info('Iniciando construcción del payload JSON para TC-DIAG');

    // Si el estudio no es craneal, detener destinos para no enviar solicitudes innecesarias.
    var esTcCraneal = String(channelMap.get('es_tc_craneal') || 'false');
    if (esTcCraneal === 'false') {
        logger.info('El mensaje no corresponde a TC craneal. Se omiten todos los destinos.');
        destinationSet.removeAll();
        return;
    }

    // Recuperar campos previamente extraídos.
    var reportId = String(channelMap.get('report_id') || '');
    var hallazgos = String(channelMap.get('hallazgos') || '');
    var opinion = String(channelMap.get('opinion') || '');
    var tecnica = String(channelMap.get('tecnica') || '');
    var datosClinicos = String(channelMap.get('datos_clinicos') || '');

    // Asegurar que hallazgos tenga un valor utilizable.
    if (!hallazgos || hallazgos.trim() === '') {
        logger.warn('El campo hallazgos llegó vacío; se usará cadena vacía como respaldo.');
        hallazgos = '';
    }

    // Construir el objeto que será serializado a JSON.
    var payload = {
        report_id: reportId,
        hallazgos: hallazgos,
        opinion: opinion,
        tecnica: tecnica,
        datos_clinicos: datosClinicos
    };

    var jsonPayload = JSON.stringify(payload);
    channelMap.put('jsonPayload', jsonPayload);

    // Registrar un resumen corto del payload para facilitar depuración clínica.
    var hallazgosLog = hallazgos.length > 100 ? hallazgos.substring(0, 100) + '...' : hallazgos;
    logger.info('Payload construido para report_id=' + reportId + ' | hallazgos=' + hallazgosLog);
} catch (error) {
    logger.error('Error al construir el JSON de salida: ' + error);
    throw error;
}
