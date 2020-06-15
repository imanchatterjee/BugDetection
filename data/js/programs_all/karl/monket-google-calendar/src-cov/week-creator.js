/* automatically generated by JSCoverage - do not edit */
if (typeof _$jscoverage === 'undefined') _$jscoverage = {};
if (! _$jscoverage['week-creator.js']) {
  _$jscoverage['week-creator.js'] = [];
  _$jscoverage['week-creator.js'][1] = 0;
  _$jscoverage['week-creator.js'][2] = 0;
  _$jscoverage['week-creator.js'][3] = 0;
  _$jscoverage['week-creator.js'][4] = 0;
  _$jscoverage['week-creator.js'][5] = 0;
  _$jscoverage['week-creator.js'][6] = 0;
  _$jscoverage['week-creator.js'][8] = 0;
  _$jscoverage['week-creator.js'][9] = 0;
  _$jscoverage['week-creator.js'][10] = 0;
  _$jscoverage['week-creator.js'][11] = 0;
  _$jscoverage['week-creator.js'][12] = 0;
  _$jscoverage['week-creator.js'][13] = 0;
  _$jscoverage['week-creator.js'][14] = 0;
  _$jscoverage['week-creator.js'][16] = 0;
  _$jscoverage['week-creator.js'][17] = 0;
  _$jscoverage['week-creator.js'][20] = 0;
  _$jscoverage['week-creator.js'][21] = 0;
  _$jscoverage['week-creator.js'][22] = 0;
  _$jscoverage['week-creator.js'][23] = 0;
  _$jscoverage['week-creator.js'][25] = 0;
  _$jscoverage['week-creator.js'][27] = 0;
  _$jscoverage['week-creator.js'][28] = 0;
  _$jscoverage['week-creator.js'][29] = 0;
  _$jscoverage['week-creator.js'][30] = 0;
  _$jscoverage['week-creator.js'][31] = 0;
  _$jscoverage['week-creator.js'][32] = 0;
  _$jscoverage['week-creator.js'][33] = 0;
  _$jscoverage['week-creator.js'][35] = 0;
  _$jscoverage['week-creator.js'][36] = 0;
  _$jscoverage['week-creator.js'][39] = 0;
  _$jscoverage['week-creator.js'][40] = 0;
  _$jscoverage['week-creator.js'][41] = 0;
  _$jscoverage['week-creator.js'][42] = 0;
  _$jscoverage['week-creator.js'][43] = 0;
}
_$jscoverage['week-creator.js'][1]++;
(function () {
  _$jscoverage['week-creator.js'][2]++;
  MKT.WeekCreator = (function (config, dayHighlighter, eventLoader) {
  _$jscoverage['week-creator.js'][3]++;
  this.config = config;
  _$jscoverage['week-creator.js'][4]++;
  this.dayHighlighter = dayHighlighter;
  _$jscoverage['week-creator.js'][5]++;
  this.eventLoader = eventLoader;
  _$jscoverage['week-creator.js'][6]++;
  return this;
});
  _$jscoverage['week-creator.js'][8]++;
  MKT.WeekCreator.prototype.create = (function (weekStart) {
  _$jscoverage['week-creator.js'][9]++;
  var _a, _b, index, label, week;
  _$jscoverage['week-creator.js'][10]++;
  week = $("#templates .week").clone().attr("id", this.config.weekIdPrefix + weekStart.customFormat(this.config.dateFormat));
  _$jscoverage['week-creator.js'][11]++;
  week.css("opacity", 0.3);
  _$jscoverage['week-creator.js'][12]++;
  $("td", week).attr("id", (function (__this) {
  _$jscoverage['week-creator.js'][13]++;
  var __func = (function (index) {
  _$jscoverage['week-creator.js'][14]++;
  return this.config.dayIdPrefix + weekStart.addDays(index).customFormat(this.config.dateFormat);
});
  _$jscoverage['week-creator.js'][16]++;
  return (function () {
  _$jscoverage['week-creator.js'][17]++;
  return __func.apply(__this, arguments);
});
})(this));
  _$jscoverage['week-creator.js'][20]++;
  _a = $(".day-label", week);
  _$jscoverage['week-creator.js'][21]++;
  for (index = 0, _b = _a.length; index < _b; index++) {
    _$jscoverage['week-creator.js'][22]++;
    label = _a[index];
    _$jscoverage['week-creator.js'][23]++;
    this.add_day_label(label, index, weekStart);
}
  _$jscoverage['week-creator.js'][25]++;
  return week;
});
  _$jscoverage['week-creator.js'][27]++;
  MKT.WeekCreator.prototype.add_day_label = (function (label, index, weekStart) {
  _$jscoverage['week-creator.js'][28]++;
  var dayDate, dayNumber;
  _$jscoverage['week-creator.js'][29]++;
  dayDate = weekStart.addDays(index);
  _$jscoverage['week-creator.js'][30]++;
  dayNumber = dayDate.customFormat("#D#");
  _$jscoverage['week-creator.js'][31]++;
  $(label).html(dayNumber);
  _$jscoverage['week-creator.js'][32]++;
  if (dayNumber === "1") {
    _$jscoverage['week-creator.js'][33]++;
    this.add_month_label(dayDate, label);
  }
  _$jscoverage['week-creator.js'][35]++;
  if (dayDate.customFormat(this.config.dateFormat) === new Date().customFormat(this.config.dateFormat)) {
    _$jscoverage['week-creator.js'][36]++;
    return this.dayHighlighter.highlightDay($(label).parent());
  }
});
  _$jscoverage['week-creator.js'][39]++;
  MKT.WeekCreator.prototype.add_month_label = (function (dayDate, label) {
  _$jscoverage['week-creator.js'][40]++;
  var monthLabel;
  _$jscoverage['week-creator.js'][41]++;
  monthLabel = $("#templates .month-label").clone().html(dayDate.customFormat("#MMMM#"));
  _$jscoverage['week-creator.js'][42]++;
  $(label).after(monthLabel);
  _$jscoverage['week-creator.js'][43]++;
  return $(label).parent().addClass("start-month");
});
})();
_$jscoverage['week-creator.js'].source = ["(function(){","  MKT.WeekCreator = function(config, dayHighlighter, eventLoader) {","    this.config = config;","    this.dayHighlighter = dayHighlighter;","    this.eventLoader = eventLoader;","    return this;","  };","  MKT.WeekCreator.prototype.create = function(weekStart) {","    var _a, _b, index, label, week;","    week = $(\"#templates .week\").clone().attr(\"id\", this.config.weekIdPrefix + weekStart.customFormat(this.config.dateFormat));","    week.css('opacity', 0.3);","    $(\"td\", week).attr(\"id\", (function(__this) {","      var __func = function(index) {","        return this.config.dayIdPrefix + weekStart.addDays(index).customFormat(this.config.dateFormat);","      };","      return (function() {","        return __func.apply(__this, arguments);","      });","    })(this));","    _a = $('.day-label', week);","    for (index = 0, _b = _a.length; index &lt; _b; index++) {","      label = _a[index];","      this.add_day_label(label, index, weekStart);","    }","    return week;","  };","  MKT.WeekCreator.prototype.add_day_label = function(label, index, weekStart) {","    var dayDate, dayNumber;","    dayDate = weekStart.addDays(index);","    dayNumber = dayDate.customFormat(\"#D#\");","    $(label).html(dayNumber);","    if (dayNumber === \"1\") {","      this.add_month_label(dayDate, label);","    }","    if (dayDate.customFormat(this.config.dateFormat) === new Date().customFormat(this.config.dateFormat)) {","      return this.dayHighlighter.highlightDay($(label).parent());","    }","  };","  MKT.WeekCreator.prototype.add_month_label = function(dayDate, label) {","    var monthLabel;","    monthLabel = $(\"#templates .month-label\").clone().html(dayDate.customFormat(\"#MMMM#\"));","    $(label).after(monthLabel);","    return $(label).parent().addClass(\"start-month\");","  };","","})();"];