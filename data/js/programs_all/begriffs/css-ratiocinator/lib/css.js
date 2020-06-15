/* global exports, require, jQuery, OBJ, CSSFontFaceRule */
/* exported CSS */

if (typeof require !== 'undefined') {
  var _ = require('../vendor/underscore-1.4.2.js'),
    $   = require('../vendor/jquery-1.8.2.js');
} else {
  var $ = jQuery.noConflict();
}

var CSS = (function () {
  'use strict';
  var my = {};

  function computedCssProperties(elt, pseudoclass) {
    var result = {}, i, j, prop, val,
      rules = window.getMatchedCSSRules(elt, pseudoclass),
      camelize = function (s) {
        return s.replace(
          /-([a-z])/g,
          function (g) { return g[1].toUpperCase(); }
        );
      };

    // dealing with "array-like" objects so cannot use underscore
    if (rules) {
      for (i = 0; i < rules.length; i += 1) {
        for (j = rules[i].style.length - 1; j >= 0; j -= 1) {
          prop = rules[i].style[j];
          val = rules[i].style[camelize(prop)];
          if (val !== 'initial') {
            result[prop] = val;
          }
        }
      }
    }
    return result;
  }

  function commonStyle(nodes) {
    return OBJ.intersection(_.map(nodes, function (k) { return $(k).data('style') || {}; }));
  }

  function liftHeritable(node) {
    node.children().each(function () { liftHeritable($(this)); });

    var heritable = [
      'cursor', 'font-family', 'font-weight', 'font-stretch', 'font-style',
      'font-size', 'font-size-adjust', 'font', 'font-synthesis', 'font-kerning',
      'font-variant-ligatures', 'font-variant-position', 'font-variant-caps',
      'font-variant-numeric', 'font-variant-alternatives', 'font-variant-east-asian',
      'font-variant', 'font-feature-settings', 'font-language-override', 'text-transform',
      'white-space', 'tab-size', 'line-break', 'word-break', 'hyphens', 'word-wrap',
      'overflow-wrap', 'text-align', 'text-align-last', 'text-justify', 'word-spacing',
      'letter-spacing', 'text-indent', 'hanging-punctuation', 'text-decoration-skip',
      'text-underline-skip', 'text-emphasis-style', 'text-emphasis-color', 'text-emphasis',
      'text-emphasis-position', 'text-shadow', 'color', 'border-collapse', 'border-spacing',
      'caption-side', 'direction', 'elevation', 'empty-cells', 'line-height', 'list-style-image',
      'list-style-position', 'list-style-type', 'list-style', 'orphans', 'pitch-range',
      'pitch', 'quotes', 'richness', 'speak-header', 'speak-numeral', 'speak-punctuation',
      'speak', 'speech-rate', 'stress', 'visibility', 'voice-family', 'volume', 'widows'
    ];

    node.children().each(function (i, kid) {
      var common = _.pick(commonStyle([node, $(kid)]), heritable);
      $(kid).data('style', OBJ.difference($(kid).data('style'), common));
    });
  }

  function stripIrrelevantStyles(node) {
    var style = node.data('style');
    if (style) {
      // remove border-width default of 0px unless it matters
      if (!style['border-style'] &&
          !style['border-bottom-style'] &&
          !style['border-left-style'] &&
          !style['border-right-style'] &&
          style['border-width'] === '0px') {
        delete style['border-width'];
      }
    }
    node.children().each(function () {
      stripIrrelevantStyles($(this));
    });
  }

  function selectorsUsed(node, soFar) {
    var tag = node.prop('tagName').toLowerCase();
    if (tag === 'head' || tag === 'script') {
      return {};
    }
    soFar = soFar || {};
    soFar[tag] = true;
    if (tag !== 'html') { // disregard classes on root element
      if (node.attr('class')) {
        _.each(node.attr('class').split(/\s+/), function (klass) {
          // Add classes but not crazy autogenerated ones that have long numbers
          if (klass && !klass.match(/\d\d\d/)) {
            soFar['.' + klass] = true;
            soFar[node.prop('tagName').toLowerCase() + '.' + klass] = true;
          }
        });
      }
      // Add ids but not crazy autogenerated ones that have long numbers
      if (node.attr('id') && !node.attr('id').match(/\d\d\d/)) {
        soFar['#' + node.attr('id')] = true;
      }
    }
    node.children().each(function () {
      selectorsUsed($(this), soFar);
    });
    return soFar;
  }

  function originatingSelectors(node, maxDepth) {
    var escapeSelector = function (sel) { return sel.replace(/([:%])/g, '\\$1'); },
      selectors        = _.map(_.keys(selectorsUsed(node)), escapeSelector),
      notId            = function (selector) { return selector[0] !== '#'; },
      subSelectors     = _.filter(selectors, notId),
      buildSelector    = function (tuple) { return tuple.join(' '); },
      addIfMatches     = function (selector) {
        if (node.find(selector).length) {
          selectors.push(selector);
        }
      };

    while (maxDepth > 1) {
      _.each(
        _.map(
          OBJ.cartesianProduct(selectors, subSelectors),
          buildSelector
        ),
        addIfMatches
      );
      maxDepth -= 1;
    }
    return selectors;
  }

  function importance(root, choice) {
    var number = root.andSelf().find(choice.selector).length,
      size = _.keys(choice.style).length,
      selectorLength = (choice.selector.match(/ /) || [1]).length,
      result = -(number * number * size);
    if (selectorLength > 1) {
      result += 1; // favor shallow selectors
    }
    return result;
  }

  function abbreviate(style) {
    // combine directional styles if possible
    _.each(
      [
        'border-TRBL-color', 'border-TRBL-style', 'border-TRBL-width',
        'padding-TRBL', 'margin-TRBL', 'border-TB-LR-radius', 'overflow-XY',
        'background-repeat-XY'
      ],
      function (template) {
        var group, values = [];
        if (template.match(/TRBL/)) {
          group = _.map(['top', 'right', 'bottom', 'left'], function (direction) {
            return template.replace('TRBL', direction);
          });
        } else if (template.match(/TB-LR/)) {
          group = _.map(
            OBJ.cartesianProduct(['top', 'bottom'], ['left', 'right']),
            function (pair) {
              return template.replace('TB', pair[0]).replace('LR', pair[1]);
            }
          );
        } else {
          group = _.map(['x', 'y'], function (direction) {
            return template.replace('XY', direction);
          });
        }
        _.each(group, function (prop) {
          if (style[prop] !== undefined) {
            values.push(style[prop]);
          }
        });
        if (values.length === group.length && _.uniq(values).length === 1) { // if all directions agree
          _.each(group, function (prop) { // then erase the directional properties
            delete style[prop];
          });
          style[template.replace(/-TRBL|TB-LR-|-XY/, '')] = values.pop();
        }
      }
    );

    // remove superfluous style
    if (!_.has(style, 'outline-style') || style['outline-style'] === 'none') {
      delete style['outline-color'];
    }
    if (!_.has(style, 'border-style') || style['border-style'] === 'none') {
      delete style['border-color'];
    }
    return style;
  }

  my.fontsUsed = function (node, soFar) {
    var fonts = node.data('style')['font-family'];
    soFar = soFar || {};
    if (fonts) {
      _.each(fonts.split(', '), function (font) {
        soFar[font] = true;
      });
    }
    node.children().each(function () {
      my.fontsUsed($(this), soFar);
    });
    return soFar;
  };

  my.mediaBreakpoints = function () {
    var inclusivity = [],
      points = _.reduce(
        my.mediaQueries(),
        function (list, q) {
          return list.concat(_.compact(
            _.map(
              ['min-width', 'max-width'],
              function (prop) {
                var found = +(q.match(prop + '\\s*:\\s*(\\d+)') || [])[1];
                if (found) { // side-effect of map
                  if (_.isUndefined(inclusivity[found])) { inclusivity[found] = {}; }
                  inclusivity[found][(prop === 'max-width') ? 'left' : 'right'] = true;
                }
                return found;
              }
            )
          ));
        },
        [0]
      );
    inclusivity[0]        = {right: true};
    inclusivity[Infinity] = {left: true};
    return {
      points: _.uniq(points.sort(function (a, b) { return a - b; }), true),
      inclusivity: inclusivity
    };
  };

  my.fontDeclarations = function () {
    $('html').find('*').andSelf().each(function (i, elt) {
      $(elt).data('style', abbreviate(computedCssProperties(elt)));
    });
    var cssRules = _.flatten(_.map(_.pluck(document.styleSheets, 'cssRules'), _.toArray)),
      fontRules = _.filter(cssRules, function (r) { return r.constructor === CSSFontFaceRule; });
    return _.pluck(fontRules, 'cssText'); // TODO: filter by fontsUsed()
  };

  my.simplerStyle = function (root) {
    root = root || $('html');

    var selectors = {};

    /* Computing styles... */
    root.find('*').andSelf().each(function (i, elt) {
      $(elt).data('style', abbreviate(computedCssProperties(elt)));
    });

    /* Lifting heritable styles... */
    liftHeritable(root);

    /* Stripping default styles... */
    stripIrrelevantStyles(root);

    /* Consolidating styles... */
    selectors = originatingSelectors(root, 2).concat(['*']);
    return consolidateStyles(root, selectors);
  };

  // recursively divide and conquer in a series of tournaments
  function consolidateStyles(root, selectors, result) {
    result = result || {};

    var selectorWithCommonStyle = function (sel) {
        return { selector: sel, style: commonStyle(root.andSelf().find(sel)) };
      },
      markImportance = function (res) {
        res.importance = importance(root, res);
        return res;
      },
      irrelevant = function (res) {
        return importance(root, res) === 0;
      },
      removeStyle = function (style) {
        return function (i, elt) {
          $(elt).data('style', OBJ.difference($(elt).data('style'), style));
        };
      };

    if (!_.isEmpty(selectors)) {
      // fight!
      var common = _.sortBy(
        _.map(
          _.map(selectors, selectorWithCommonStyle),
          markImportance
        ),
        'importance'
      );

      // Record the victor
      var best = common.shift();
      if (!_.isEmpty(best.style)) {
        result[best.selector] = best.style;
        root.andSelf().find(best.selector).each(removeStyle(best.style));
      }

      // Split the sorted remainder into stronger and weaker groups
      selectors = _.pluck(
        _.reject(common, irrelevant),
        'selector'
      );

      var middle = Math.ceil(selectors.length/2),
        greater = selectors.slice(0, middle), lesser = selectors.slice(middle);

      // Run tournaments in each group recursively. They all share "result"
      consolidateStyles(root, greater, result);
      consolidateStyles(root, lesser, result);
    }
    return result;
  }

  my.renderStyle = function (properties, selector, indentLevel) {
    var css = '', pad = '';
    _(indentLevel || 0).times(function () { pad += '\t'; });

    if (!_.isEmpty(properties)) {
      css += (pad + selector + ' {\n');
      _.each(properties, function (val, key) {
        css += (pad + '\t' + key + ': ' + val + ';\n');
      });
      css += (pad + '}\n');
    }
    return css;
  };

  my.mediaQueries = function () {
    return _.pluck(
      _.flatten(_.map(
        _.pluck(document.styleSheets, 'cssRules'),
        function (rule) {
          return _.compact(_.pluck(rule, 'media'));
        }
      )),
      'mediaText'
    );
  };

  my.mediaWidthIntervals = function () {
    var breaks  = my.mediaBreakpoints(),
      intervals = _.filter(
        _.zip(breaks.points, breaks.points.slice(1).concat([Infinity])),
        function (I) { return I[1] - I[0] > 1; }
      );
    return _.map(
      intervals,
      function (I) {
        var min = I[0], max = I[1];
        return {
          min:    breaks.inclusivity[min].right ? min : min + 1,
          max:    breaks.inclusivity[max].left  ? max : max - 1,
          sample: max !== Infinity ? min + Math.round((max - min) / 2) : min + 1
        };
      }
    );
  };

  if (typeof exports !== 'undefined') {
    _.extend(exports, my);
  }
  return my;
}());