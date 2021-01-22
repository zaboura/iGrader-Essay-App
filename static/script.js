$('#submit-form').submit(function(e) {
    e.preventDefault();
    $form = $(this)
    var formData = new FormData(this);
    $.ajax({

        url: "predict/",
        type: 'POST',
        data: formData,

        success: function(json) {
            response = json;

            if (response['type'] == 'file_error') {

                file_error()

            } else if (response['type'] == 'word_limit') {

                word_limit_upload()

            } else {


                setTimeout(values_feed, 255);


                console.log(json['type']);


                document.getElementById("type").innerHTML = json['type'];


                $("#loader").hide();


                $("#div-info").hide();


                $("#input_text_div").fadeOut(250);


                $("#summary").delay(250).fadeIn(250);


            };





        },


        error: function(xhr, errmsg, err) { // add the error to the dom


            console.log(xhr.status + ": " + xhr.responseText);


            window.location.href = "/error"; // provide a bit more info about the error to the console


        },


        cache: false,


        contentType: false,


        processData: false


    });


});