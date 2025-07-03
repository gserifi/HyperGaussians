// window.HELP_IMPROVE_VIDEOJS = false;

// var INTERP_BASE = "./static/interpolation/stacked";
// var NUM_INTERP_FRAMES = 240;

// var interp_images = [];
// function preloadInterpolationImages() {
//   for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
//     var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
//     interp_images[i] = new Image();
//     interp_images[i].src = path;
//   }
// }

// function setInterpolationImage(i) {
//   var image = interp_images[i];
//   image.ondragstart = function() { return false; };
//   image.oncontextmenu = function() { return false; };
//   $('#interpolation-image-wrapper').empty().append(image);
// }

window.STATIC_ASSET_SERVER = 'https://files.ait.ethz.ch/projects/cafca/web/static/';


function updateSrcwithStaticAssets(className) {
    // Select all elements with the given class name
    const elements = document.getElementsByClassName(className);

    // Iterate through the selected elements
    for (let i = 0; i < elements.length; i++) {
        // Check if the element has a src attribute
        if ('src' in elements[i]) {
            // Change the src attribute to the new value
            console.log(elements[i].src);
            let idx = elements[i].src.indexOf("ASSETS/");
            if (idx > 0) {
                elements[i].src = window.STATIC_ASSET_SERVER + elements[i].src.slice(idx + 7);

                if (elements[i].parentElement.nodeName === "VIDEO"){
                    elements[i].parentElement.load();
                }
            }
        }
    }
}


$(document).ready(function() {
    updateSrcwithStaticAssets('ait-asset');
    // Check for click events on the navbar burger icon

    // $(".navbar-burger").click(function() {
    //   // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    //   $(".navbar-burger").toggleClass("is-active");
    //   $(".navbar-menu").toggleClass("is-active");
    //
    // });
    //
    // var options = {
	// 		slidesToScroll: 1,
	// 		slidesToShow: 3,
	// 		loop: true,
	// 		infinite: true,
	// 		autoplay: false,
	// 		autoplaySpeed: 3000,
    // }
    //
	// 	// Initialize all div with carousel class
    // var carousels = bulmaCarousel.attach('.carousel', options);
    //
    // // Loop on each carousel initialized
    // for(var i = 0; i < carousels.length; i++) {
    // 	// Add listener to  event
    // 	carousels[i].on('before:show', state => {
    // 		console.log(state);
    // 	});
    // }
    //
    // // Access to bulmaCarousel instance of an element
    // var element = document.querySelector('#my-element');
    // if (element && element.bulmaCarousel) {
    // 	// bulmaCarousel instance is available as element.bulmaCarousel
    // 	element.bulmaCarousel.on('before-show', function(state) {
    // 		console.log(state);
    // 	});
    // }
    //
    // /*var player = document.getElementById('interpolation-video');
    // player.addEventListener('loadedmetadata', function() {
    //   $('#interpolation-slider').on('input', function(event) {
    //     console.log(this.value, player.duration);
    //     player.currentTime = player.duration / 100 * this.value;
    //   })
    // }, false);*/
    // preloadInterpolationImages();
    //
    // $('#interpolation-slider').on('input', function(event) {
    //   setInterpolationImage(this.value);
    // });
    // setInterpolationImage(0);
    // $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
    //
    // bulmaSlider.attach();

})
