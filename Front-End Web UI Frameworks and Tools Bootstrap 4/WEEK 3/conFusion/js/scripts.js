$(document).ready(()=>{

    $("#carouselButton").click(function(){
           
        if ($("#carouselButton").children("span").hasClass('fa-pause')) {
            console.log("HERE IF")
            $("#mycarousel").carousel('pause');
            console.log("HERE IF")
            $("#carouselButton").children("span").removeClass('fa-pause');
            $("#carouselButton").children("span").addClass('fa-play');
        }
        else if ($("#carouselButton").children("span").hasClass('fa-play')){
            $("#mycarousel").carousel('cycle');
            console.log("HERE ELSE")
            $("#carouselButton").children("span").removeClass('fa-play');
            $("#carouselButton").children("span").addClass('fa-pause');                    
        }
    });

    $("#loginButton").click(function(){
        $('#loginModal').modal('show');
    });

    $("#reserveButton").click(function(){
        $('#reserveModal').modal('show');
    });

});