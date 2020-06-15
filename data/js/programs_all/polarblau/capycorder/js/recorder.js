// Generated by CoffeeScript 1.3.3
(function() {
  var blurFocused, disableHighlighting, enableHighlighting, highlighter, init;

  init = function() {
    var recorder, ui, _ref;
    _ref = [
      new RecorderUI({
        chrome: chrome
      }), null
    ], ui = _ref[0], recorder = _ref[1];
    chrome.extension.sendRequest({
      name: 'loaded'
    });
    chrome.extension.onRequest.addListener(function(request) {
      var recorderOptions, state;
      switch (request.name) {
        case 'stateChanged':
          state = request.state;
          recorderOptions = {
            scope: document,
            afterCapture: function(dataAsJSON) {
              return chrome.extension.sendRequest({
                name: 'captured',
                data: dataAsJSON
              });
            }
          };
          switch (state) {
            case 'name':
              ui.showNamePrompt(function(name) {
                return chrome.extension.sendRequest({
                  name: 'named',
                  specsName: name
                });
              });
              break;
            case 'capture.actions':
              recorder = new Capybara.Recorders.Actions(recorderOptions);
              recorder.start();
              break;
            case 'capture.matchers':
              blurFocused();
              if (recorder != null) {
                recorder.stop();
              }
              recorder = new Capybara.Recorders.Matchers(recorderOptions);
              recorder.start();
              enableHighlighting();
              break;
            case 'generate':
              if (recorder != null) {
                recorder.stop();
              }
              disableHighlighting();
          }
          if (state !== 'off' && state !== 'name') {
            return ui.show(state);
          }
      }
    });
    return window.onbeforeunload = blurFocused;
  };

  blurFocused = function() {
    $('input:focus').blur();
    return null;
  };

  highlighter = null;

  enableHighlighting = function() {
    var _this = this;
    highlighter || (highlighter = new SelectionBox);
    return $(document).on('mousemove.highlight', function(e) {
      e.preventDefault();
      e.stopPropagation();
      $('body').css('cursor', 'crosshair');
      return highlighter.highlight(e.target);
    });
  };

  disableHighlighting = function() {
    if (highlighter != null) {
      $(document).off('mousemove.highlight');
      $('body').css('cursor', '');
      return highlighter.hide();
    }
  };

  $(document).ready(init);

}).call(this);