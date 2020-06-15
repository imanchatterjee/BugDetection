// Generated by CoffeeScript 1.4.0
(function() {
  var $,
    __hasProp = {}.hasOwnProperty;

  $ = OpenSpending.$;

  OpenSpending.Browser = (function() {

    Browser.prototype.options = {
      source: '',
      table: {},
      facets: {}
    };

    function Browser(element, dataset, options) {
      this.dataset = dataset;
      this.element = $(element);
      this.options = $.extend(true, {}, this.options, options);
      this.req = $.getJSON(this.options.source + '/' + this.dataset + '/dimensions.json');
      this._buildTable();
      this._buildFacets();
      this._loadRoute();
    }

    Browser.prototype.init = function() {
      var _this = this;
      return this.req.then(function(data) {
        var d, facetDimensions, k, _i, _j, _len, _len1, _ref;
        _this.dimensions = {};
        for (_i = 0, _len = data.length; _i < _len; _i++) {
          d = data[_i];
          _this.dimensions[d.key] = d;
        }
        _this.table.addColumn({
          name: 'time.year',
          label: _this.dimensions['time'].label
        });
        _ref = ['from', 'to'];
        for (_j = 0, _len1 = _ref.length; _j < _len1; _j++) {
          d = _ref[_j];
          if (_this.dimensions[d] != null) {
            _this.table.addColumn({
              name: "" + d,
              label: _this.dimensions[d].label
            });
          }
        }
        _this.table.addColumn({
          name: 'amount',
          label: 'Amount',
          data: function(data) {
            return OpenSpending.Utils.formatAmountWithCommas(data.amount || 0);
          }
        });
        _this.table.addColumn({
          data: function(data) {
            return "<a href='" + data.html_url + "'>details&raquo;</a>";
          },
          sortable: false
        });
        facetDimensions = (function() {
          var _ref1, _results;
          _ref1 = this.dimensions;
          _results = [];
          for (k in _ref1) {
            if (!__hasProp.call(_ref1, k)) continue;
            d = _ref1[k];
            if (d.facet) {
              _results.push(d);
            } else {
              continue;
            }
          }
          return _results;
        }).call(_this);
        _this.faceter.setDimensions(facetDimensions);
        _this.table.init();
        _this.faceter.init();
        return _this.element.trigger('browser:init');
      });
    };

    Browser.prototype.addFilter = function(key, value) {
      this.faceter.addFilter(key, value);
      this.table.addFilter(key, value);
      return this._updateRoute();
    };

    Browser.prototype.removeFilter = function(key) {
      this.faceter.removeFilter(key);
      this.table.removeFilter(key);
      return this._updateRoute();
    };

    Browser.prototype.redraw = function() {
      this.faceter.redraw();
      return this.table.redraw();
    };

    Browser.prototype._updateRoute = function() {
      var hash, k, v;
      hash = ((function() {
        var _ref, _results;
        _ref = this.faceter.filters;
        _results = [];
        for (k in _ref) {
          if (!__hasProp.call(_ref, k)) continue;
          v = _ref[k];
          _results.push("" + k + ":" + (encodeURIComponent(v)));
        }
        return _results;
      }).call(this)).join("/");
      return window.location.hash = hash;
    };

    Browser.prototype._loadRoute = function() {
      var filterValue, filters, hash, key, val, value, _i, _len, _results;
      hash = window.location.hash.substr(1);
      if (hash === "") {
        return;
      }
      filters = (function() {
        var _i, _len, _ref, _results;
        _ref = hash.split("/");
        _results = [];
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          val = _ref[_i];
          _results.push(val.split(":"));
        }
        return _results;
      })();
      _results = [];
      for (_i = 0, _len = filters.length; _i < _len; _i++) {
        filterValue = filters[_i];
        key = filterValue[0];
        value = decodeURIComponent(filterValue[1]);
        _results.push(this.addFilter(key, value));
      }
      return _results;
    };

    Browser.prototype._buildTable = function() {
      var options, tableEl;
      tableEl = this.element.find('.browser_datatable')[0];
      if (tableEl.length === 0) {
        tableEl = $('<div class="browser_datatable"></div>').appendTo(this.element);
      }
      options = $.extend(true, {
        source: this.options.source + '/api/2/search',
        sorting: [['amount', 'desc']],
        defaultParams: {
          dataset: this.dataset
        },
        tableOptions: {
          sDom: "<'row'<'span0'l><'span9'f>r>t<'row'<'span4'i><'span5'p>>",
          sPaginationType: "bootstrap"
        }
      }, this.options.table);
      return this.table = new OpenSpending.DataTable(tableEl, options);
    };

    Browser.prototype._buildFacets = function() {
      var facetEl, options,
        _this = this;
      facetEl = this.element.find('.browser_faceter');
      if (facetEl.length === 0) {
        facetEl = $('<div class="browser_faceter"></div>').appendTo(this.element);
      }
      options = $.extend(true, {
        source: this.options.source + '/api/2/search',
        defaultParams: {
          dataset: this.dataset,
          expand_facet_dimensions: true
        }
      }, this.options.faceter);
      this.faceter = new OpenSpending.Faceter(facetEl, [], options);
      this.faceter.element.off('faceter:addFilter');
      this.faceter.element.off('faceter:removeFilter');
      this.faceter.element.on('faceter:addFilter', function(e, k, v) {
        _this.addFilter(k, v, false);
        return _this.redraw();
      });
      return this.faceter.element.on('faceter:removeFilter', function(e, k) {
        _this.removeFilter(k, false);
        return _this.redraw();
      });
    };

    return Browser;

  })();

}).call(this);