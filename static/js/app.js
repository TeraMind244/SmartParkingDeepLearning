function doAjax(url, method, successCallback, errorCallback) {
    $.ajax({
        url: url,
        dataType: 'json',
        type: method,
        success: function(response, status) {
            successCallback(response, status);
        },
        error: function (XMLHttpRequest, textStatus, errorThrown) {
            if (errorCallback) {
                errorCallback(XMLHttpRequest, textStatus, errorThrown);
            }
        }
    });
}

function getDataWithAjax(id, url, method) {
    $.ajax({
        url: url,
        type: method,
        success: function(response, status) {
            $("#" + id).html(response);
        },
        error: function (XMLHttpRequest, textStatus, errorThrown) {
            // do nothing on error
        }
    });
}