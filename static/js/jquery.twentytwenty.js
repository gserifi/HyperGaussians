(function($) {

    $.fn.twentytwenty = function(options) {
        var options = $.extend({
            default_offset_pct_x: 0.5,
            default_offset_pct_y: 0.5,
            click_needed: false,
            placeholder: false,
        }, options);
        var render = function() {
            var self = this;

            var withPlaceholder = function(img, requestReload) {
                if (img.src == options.placeholder) {
                    return false;
                }
                if (options.placeholder && (!img.complete || img.naturalWidth == 0)) {
                    if (requestReload) {
                        img.onload = render.bind(self);
                    }
                    img.src = options.placeholder;
                    return requestReload;
                }
                return false;
            };

            var sliderPctX = options.default_offset_pct_x;
            var sliderPctY = options.default_offset_pct_y;
            var container = $(this);
            var clickNeeded = options.click_needed;


            var numImgs = container.children("img").length;
            var imgs = [container.children("img:eq(0)"),
                        container.children("img:eq(1)"),
                        container.children("img:eq(2)"),
                        container.children("img:eq(3)")];

            var temp = 0;
            var referenceIndex = 0;
            var success = true;
            for (var i = 0; i < numImgs; i++)
            {
                // Reload the with the first failed image
                success = !withPlaceholder(imgs[i].get(0), success);
                if (temp < imgs[i].get(0).naturalWidth)
                {
                    temp = imgs[i].get(0).naturalWidth;
                    referenceIndex = i;
                }
            }

            if (!success) {
                return;
            }

            container.css("max-width", imgs[referenceIndex].get(0).naturalWidth);
            container.addClass("twentytwenty-compare-" + numImgs)

            for (var i = 0; i < numImgs; i++)
                imgs[i].addClass("twentytwenty-" + (i+1));

            container.append("<div class='twentytwenty-overlay'></div>");
            var overlay = container.find(".twentytwenty-overlay");


            for (var i = 0; i < numImgs; i++)
                overlay.append("<div class='twentytwenty-label-" + (i+1) + "'>" + imgs[i].attr('alt') + "</div>");

            var labels = [overlay.find(".twentytwenty-label-1"),
                          overlay.find(".twentytwenty-label-2"),
                          overlay.find(".twentytwenty-label-3"),
                          overlay.find(".twentytwenty-label-4")];

            for (var i = 0; i < numImgs; i++)
                overlay.append("<div class='twentytwenty-frame-" + (i+1) + "'></div>");

            var frames = [overlay.find(".twentytwenty-frame-1"),
                          overlay.find(".twentytwenty-frame-2"),
                          overlay.find(".twentytwenty-frame-3"),
                          overlay.find(".twentytwenty-frame-4")];

            var calcOffset = function(dimensionPctX, dimensionPctY) {
                var w = imgs[referenceIndex].width();
                var h = imgs[referenceIndex].height();
                return {
                    w: w + "px",
                    h: h + "px",
                    cw: (dimensionPctX * w) + "px",
                    ch: (dimensionPctY * h) + "px",
                    w2: w,
                    h2: h,
                    cw2: (dimensionPctX * w),
                    ch2: (dimensionPctY * h)
                };
            };

            var adjustContainer = function(offset) {
                overlay.css("width", offset.w);
                overlay.css("height", offset.h);
                if (numImgs == 2)
                {
                    imgs[0].css("clip", "rect(0," + offset.cw + "," + offset.h + ",0)");

                    labels[0].css({right: offset.w2 - offset.cw2});
                    labels[1].css({left: offset.cw2});

                    frames[0].css({width: offset.cw2, height: offset.h2});
                    frames[1].css({width: offset.w2 - offset.cw2, height: offset.h2});
                }
                else if (numImgs == 3)
                {
                    imgs[0].css("clip", "rect(0," + offset.cw + "," + offset.ch + ",0)");
                    imgs[1].css("clip", "rect(0," + offset.w + "," + offset.ch + "," + offset.cw + ")");
                    imgs[2].css("clip", "rect(" + offset.ch + "," + offset.w + "," + offset.h + ",0)");

                    frames[0].css({width: offset.cw2, height: offset.ch2});
                    frames[1].css({width: offset.w2 - offset.cw2, height: offset.ch2});
                    frames[2].css({width: offset.w2, height: offset.h2 - offset.ch2});

                    labels[0].css({right: offset.w2 - offset.cw2, bottom: offset.h2 - offset.ch2});
                    labels[1].css({left: offset.cw2, bottom: offset.h2 - offset.ch2});

                    labels[2].css({top: offset.ch2});
                }
                else if (numImgs == 4)
                {
                    imgs[0].css("clip", "rect(0," + offset.cw + "," + offset.ch + ",0)");
                    imgs[1].css("clip", "rect(0," + offset.w + "," + offset.ch + "," + offset.cw + ")");
                    imgs[2].css("clip", "rect(" + offset.ch + "," + offset.cw + "," + offset.h + ",0)");
                    imgs[3].css("clip", "rect(" + offset.ch + "," + offset.w + "," + offset.h + "," + offset.cw + ")");

                    frames[0].css({width: offset.cw2, height: offset.ch2});
                    frames[1].css({width: offset.w2 - offset.cw2, height: offset.ch2});
                    frames[2].css({width: offset.cw2, height: offset.h2 - offset.ch2});
                    frames[3].css({width: offset.w2 - offset.cw2, height: offset.h2 - offset.ch2});

                    labels[0].css({right: offset.w2 - offset.cw2, bottom: offset.h2 - offset.ch2});
                    labels[1].css({left: offset.cw2, bottom: offset.h2 - offset.ch2});

                    labels[2].css({right: offset.w2 - offset.cw2, top: offset.ch2});
                    labels[3].css({left: offset.cw2, top: offset.ch2});
                }

                container.css("height", offset.h);
            };

            var adjustSlider = function(pctX, pctY) {
                var offset = calcOffset(pctX, pctY);
                adjustContainer(offset);
            }

            $(window).on("resize.twentytwenty", function(e) {

                var refRect = imgs[referenceIndex].get(0).getBoundingClientRect();
                for (var i = 0; i < numImgs; i++)
                {
                    if (i != referenceIndex)
                    {
                        imgs[i].css("width", refRect.width);
                        imgs[i].css("height", refRect.height);
                    }
                }
                adjustSlider(sliderPctX, sliderPctY);
            });

            container.on("move mousemove", function(e) {
                    sliderPctX = Math.max(0, Math.min(1, (e.pageX - container.offset().left) / imgs[referenceIndex].width()));
                    sliderPctY = Math.max(0, Math.min(1, (e.pageY - container.offset().top) / imgs[referenceIndex].height()));
                    adjustSlider(sliderPctX, sliderPctY);
            });

            container.find("img").on("mousedown", function(event) {
                event.preventDefault();
            });

            $(window).trigger("resize.twentytwenty");
        }

        return this.each(render);
    };

})(jQuery);
