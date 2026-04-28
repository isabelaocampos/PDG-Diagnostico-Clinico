// File: mirth/transformaciones/extraer_obx.js
// Script de Mirth Connect para extraer segmentos HL7 y preparar campos clínicos.
try {
    logger.info('Iniciando extracción de datos HL7 para TC-DIAG');

    // Extraer identificador del reporte desde el mensaje HL7.
    var reportId = msg['MSH']['MSH.10'].toString();
    logger.info('report_id extraído: ' + reportId);

    // Extraer el código del estudio desde OBR.4.2.
    var studyCode = '';
    if (msg['OBR'] && msg['OBR']['OBR.4'] && msg['OBR']['OBR.4']['OBR.4.2']) {
        studyCode = msg['OBR']['OBR.4']['OBR.4.2'].toString();
    }
    logger.info('study_code extraído: ' + studyCode);

    // Validar que el estudio corresponda a una tomografía de cráneo.
    var studyCodeLower = studyCode.toLowerCase();
    var esTcCraneal = (
        studyCodeLower.indexOf('craneal') >= 0 ||
        studyCodeLower.indexOf('cerebral') >= 0 ||
        studyCodeLower.indexOf('craneo') >= 0 ||
        studyCodeLower.indexOf('tc craneo') >= 0 ||
        studyCodeLower.indexOf('tac craneo') >= 0
    );

    if (!esTcCraneal) {
        logger.info('El estudio no corresponde a TC craneal. Canal marcado como falso.');
        channelMap.put('es_tc_craneal', 'false');
        return;
    }

    // Inicializar acumuladores de texto clínico.
    var hallazgos = '';
    var opinion = '';
    var tecnica = '';
    var datosClinicos = '';

    // Recorrer todos los segmentos OBX del mensaje.
    var obxSegments = msg..OBX;
    for (var i = 0; i < obxSegments.length(); i++) {
        var obx = obxSegments[i];
        var tipo = '';
        var valor = '';

        if (obx['OBX.3'] && obx['OBX.3']['OBX.3.2']) {
            tipo = obx['OBX.3']['OBX.3.2'].toString().toLowerCase();
        }
        if (obx['OBX.5']) {
            valor = obx['OBX.5'].toString();
        }

        logger.info('Procesando OBX tipo=' + tipo + ' valor=' + valor);

        if (tipo.indexOf('hallazgo') >= 0 || tipo.indexOf('finding') >= 0) {
            hallazgos += ' ' + valor;
        }
        if (tipo.indexOf('opinion') >= 0 || tipo.indexOf('impresion') >= 0 || tipo.indexOf('conclusion') >= 0) {
            opinion += ' ' + valor;
        }
        if (tipo.indexOf('tecnica') >= 0) {
            tecnica += ' ' + valor;
        }
        if (tipo.indexOf('clinico') >= 0 || tipo.indexOf('antecedente') >= 0) {
            datosClinicos += ' ' + valor;
        }
    }

    // Guardar resultados en el channelMap para los siguientes pasos del canal.
    channelMap.put('report_id', reportId);
    channelMap.put('hallazgos', String(hallazgos).trim());
    channelMap.put('opinion', String(opinion).trim());
    channelMap.put('tecnica', String(tecnica).trim());
    channelMap.put('datos_clinicos', String(datosClinicos).trim());
    channelMap.put('es_tc_craneal', 'true');

    logger.info('Extracción completada correctamente para report_id=' + reportId);
} catch (error) {
    logger.error('Error al extraer segmentos OBX: ' + error);
    throw error;
}
