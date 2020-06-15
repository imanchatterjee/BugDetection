(function(ve, dom){
	/**
	 * DOM选择器
	 **/
	var chunker = /((?:\((?:\([^()]+\)|[^()]+)+\)|\[(?:\[[^\[\]]*\]|['"][^'"]*['"]|[^\[\]'"]+)+\]|\\.|[^ >+~,(\[\\]+)+|[>+~])(\s*,\s*)?((?:.|\r|\n)*)/g,
		done = 0,
		toString = Object.prototype.toString,
		hasDuplicate = false,
		baseHasDuplicate = true,
		rBackslash = /\\/g,
		rNonWord = /\W/,
		tmpVar, rSpeedUp = /^(\w+$)|^\.([\w\-]+$)|^#([\w\-]+$)|^(\w+)\.([\w\-]+$)/;
	[0, 0].sort(function () {
		baseHasDuplicate = false;
		return 0;
	});

	var Sizzle = function (selector, context, results, seed) {
			results = results || [];
			context = context || document;
			var origContext = context;
			if (context.nodeType !== 1 && context.nodeType !== 9) {
				return [];
			}
			if (!selector || typeof selector !== "string") {
				return results;
			}
			var m, set, checkSet, extra, ret, cur, pop, i, prune = true,
				contextXML = Sizzle.isXML(context),
				parts = [],
				soFar = selector,
				speedUpMatch;
			if (!contextXML) {
				speedUpMatch = rSpeedUp.exec(selector);
				if (speedUpMatch) {
					if (context.nodeType === 1 || context.nodeType === 9) {
						if (speedUpMatch[1]) {
							return makeArray(context.getElementsByTagName(selector), results);
						} else if (speedUpMatch[2] || (speedUpMatch[4] && speedUpMatch[5])) {
							if (context.getElementsByClassName && speedUpMatch[2]) {
								return makeArray(context.getElementsByClassName(speedUpMatch[2]), results);
							} else {
								var suElems = context.getElementsByTagName(speedUpMatch[4] || '*'),
									suResBuff = [],
									suIt, suCN = ' ' + (speedUpMatch[2] || speedUpMatch[5]) + ' ';
								for (var sui = 0, sulen = suElems.length; sui < sulen; ++sui) {
									suIt = suElems[sui];
									((' ' + suIt.className + ' ').indexOf(suCN) > -1) && suResBuff.push(suIt);
								}
								return makeArray(suResBuff, results);
							}
						}
					}
					if (context.nodeType === 9) {
						if ((selector === "body" || selector.toLowerCase() === "body") && context.body) {
							return makeArray([context.body], results);
						} else if (speedUpMatch[3]) {
							return (tmpVar = context.getElementById(speedUpMatch[3])) ? makeArray([tmpVar], results) : makeArray([], results);
						}
					}
				}
			}
			do {
				chunker.exec("");
				m = chunker.exec(soFar);
				if (m) {
					soFar = m[3];
					parts.push(m[1]);
					if (m[2]) {
						extra = m[3];
						break;
					}
				}
			} while (m);
			if (parts.length > 1 && origPOS.exec(selector)) {
				if (parts.length === 2 && Expr.relative[parts[0]]) {
					set = posProcess(parts[0] + parts[1], context);
				} else {
					set = Expr.relative[parts[0]] ? [context] : Sizzle(parts.shift(), context);
					while (parts.length) {
						selector = parts.shift();
						if (Expr.relative[selector]) {
							selector += parts.shift();
						}
						set = posProcess(selector, set);
					}
				}
			} else {
				if (!seed && parts.length > 1 && context.nodeType === 9 && !contextXML && Expr.match.ID.test(parts[0]) && !Expr.match.ID.test(parts[parts.length - 1])) {
					ret = Sizzle.find(parts.shift(), context, contextXML);
					context = ret.expr ? Sizzle.filter(ret.expr, ret.set)[0] : ret.set[0];
				}
				if (context) {
					ret = seed ? {
						expr: parts.pop(),
						set: makeArray(seed)
					} : Sizzle.find(parts.pop(), parts.length === 1 && (parts[0] === "~" || parts[0] === "+") && context.parentNode ? context.parentNode : context, contextXML);
					set = ret.expr ? Sizzle.filter(ret.expr, ret.set) : ret.set;
					if (parts.length > 0) {
						checkSet = makeArray(set);
					} else {
						prune = false;
					}
					while (parts.length) {
						cur = parts.pop();
						pop = cur;
						if (!Expr.relative[cur]) {
							cur = "";
						} else {
							pop = parts.pop();
						}
						if (pop == null) {
							pop = context;
						}
						Expr.relative[cur](checkSet, pop, contextXML);
					}
				} else {
					checkSet = parts = [];
				}
			}
			if (!checkSet) {
				checkSet = set;
			}
			if (!checkSet) {
				Sizzle.error(cur || selector);
			}
			if (toString.call(checkSet) === "[object Array]") {
				if (!prune) {
					results.push.apply(results, checkSet);
				} else if (context && context.nodeType === 1) {
					for (i = 0; checkSet[i] != null; i++) {
						if (checkSet[i] && (checkSet[i] === true || checkSet[i].nodeType === 1 && Sizzle.contains(context, checkSet[i]))) {
							results.push(set[i]);
						}
					}
				} else {
					for (i = 0; checkSet[i] != null; i++) {
						if (checkSet[i] && checkSet[i].nodeType === 1) {
							results.push(set[i]);
						}
					}
				}
			} else {
				makeArray(checkSet, results);
			}
			if (extra) {
				Sizzle(extra, origContext, results, seed);
				Sizzle.uniqueSort(results);
			}
			return results;
		};
	Sizzle.uniqueSort = function (results) {
		if (sortOrder) {
			hasDuplicate = baseHasDuplicate;
			results.sort(sortOrder);
			if (hasDuplicate) {
				for (var i = 1; i < results.length; i++) {
					if (results[i] === results[i - 1]) {
						results.splice(i--, 1);
					}
				}
			}
		}
		return results;
	};
	Sizzle.matches = function (expr, set) {
		return Sizzle(expr, null, null, set);
	};
	Sizzle.matchesSelector = function (node, expr) {
		return Sizzle(expr, null, null, [node]).length > 0;
	};
	Sizzle.find = function (expr, context, isXML) {
		var set;
		if (!expr) {
			return [];
		}
		for (var i = 0, l = Expr.order.length; i < l; i++) {
			var match, type = Expr.order[i];
			if ((match = Expr.leftMatch[type].exec(expr))) {
				var left = match[1];
				match.splice(1, 1);
				if (left.substr(left.length - 1) !== "\\") {
					match[1] = (match[1] || "").replace(rBackslash, "");
					set = Expr.find[type](match, context, isXML);
					if (set != null) {
						expr = expr.replace(Expr.match[type], "");
						break;
					}
				}
			}
		}
		if (!set) {
			set = typeof context.getElementsByTagName !== "undefined" ? context.getElementsByTagName("*") : [];
		}
		return {
			set: set,
			expr: expr
		};
	};
	Sizzle.filter = function (expr, set, inplace, not) {
		var match, anyFound, old = expr,
			result = [],
			curLoop = set,
			isXMLFilter = set && set[0] && Sizzle.isXML(set[0]);
		while (expr && set.length) {
			for (var type in Expr.filter) {
				if ((match = Expr.leftMatch[type].exec(expr)) != null && match[2]) {
					var found, item, filter = Expr.filter[type],
						left = match[1];
					anyFound = false;
					match.splice(1, 1);
					if (left.substr(left.length - 1) === "\\") {
						continue;
					}
					if (curLoop === result) {
						result = [];
					}
					if (Expr.preFilter[type]) {
						match = Expr.preFilter[type](match, curLoop, inplace, result, not, isXMLFilter);
						if (!match) {
							anyFound = found = true;
						} else if (match === true) {
							continue;
						}
					}
					if (match) {
						for (var i = 0;
						(item = curLoop[i]) != null; i++) {
							if (item) {
								found = filter(item, match, i, curLoop);
								var pass = not ^ !! found;
								if (inplace && found != null) {
									if (pass) {
										anyFound = true;
									} else {
										curLoop[i] = false;
									}
								} else if (pass) {
									result.push(item);
									anyFound = true;
								}
							}
						}
					}
					if (found !== undefined) {
						if (!inplace) {
							curLoop = result;
						}
						expr = expr.replace(Expr.match[type], "");
						if (!anyFound) {
							return [];
						}
						break;
					}
				}
			}
			if (expr === old) {
				if (anyFound == null) {
					Sizzle.error(expr);
				} else {
					break;
				}
			}
			old = expr;
		}
		return curLoop;
	};
	Sizzle.error = function (msg) {
		throw "Syntax error, unrecognized expression: " + msg;
	};
	var Expr = Sizzle.selectors = {
		order: ["ID", "NAME", "TAG"],
		match: {
			ID: /#((?:[\w\u00c0-\uFFFF\-]|\\.)+)/,
			CLASS: /\.((?:[\w\u00c0-\uFFFF\-]|\\.)+)/,
			NAME: /\[name=['"]*((?:[\w\u00c0-\uFFFF\-]|\\.)+)['"]*\]/,
			ATTR: /\[\s*((?:[\w\u00c0-\uFFFF\-]|\\.)+)\s*(?:(\S?=)\s*(?:(['"])(.*?)\3|(#?(?:[\w\u00c0-\uFFFF\-]|\\.)*)|)|)\s*\]/,
			TAG: /^((?:[\w\u00c0-\uFFFF\*\-]|\\.)+)/,
			CHILD: /:(only|nth|last|first)-child(?:\(\s*(even|odd|(?:[+\-]?\d+|(?:[+\-]?\d*)?n\s*(?:[+\-]\s*\d+)?))\s*\))?/,
			POS: /:(nth|eq|gt|lt|first|last|even|odd)(?:\((\d*)\))?(?=[^\-]|$)/,
			PSEUDO: /:((?:[\w\u00c0-\uFFFF\-]|\\.)+)(?:\((['"]?)((?:\([^\)]+\)|[^\(\)]*)+)\2\))?/
		},
		leftMatch: {},
		attrMap: {
			"class": "className",
			"for": "htmlFor"
		},
		attrHandle: {
			href: function (elem) {
				return elem.getAttribute("href");
			},
			type: function (elem) {
				return elem.getAttribute("type");
			}
		},
		relative: {
			"+": function (checkSet, part) {
				var isPartStr = typeof part === "string",
					isTag = isPartStr && !rNonWord.test(part),
					isPartStrNotTag = isPartStr && !isTag;
				if (isTag) {
					part = part.toLowerCase();
				}
				for (var i = 0, l = checkSet.length, elem; i < l; i++) {
					if ((elem = checkSet[i])) {
						while ((elem = elem.previousSibling) && elem.nodeType !== 1) {}
						checkSet[i] = isPartStrNotTag || elem && elem.nodeName.toLowerCase() === part ? elem || false : elem === part;
					}
				}
				if (isPartStrNotTag) {
					Sizzle.filter(part, checkSet, true);
				}
			},
			">": function (checkSet, part) {
				var elem, isPartStr = typeof part === "string",
					i = 0,
					l = checkSet.length;
				if (isPartStr && !rNonWord.test(part)) {
					part = part.toLowerCase();
					for (; i < l; i++) {
						elem = checkSet[i];
						if (elem) {
							var parent = elem.parentNode;
							checkSet[i] = parent.nodeName.toLowerCase() === part ? parent : false;
						}
					}
				} else {
					for (; i < l; i++) {
						elem = checkSet[i];
						if (elem) {
							checkSet[i] = isPartStr ? elem.parentNode : elem.parentNode === part;
						}
					}
					if (isPartStr) {
						Sizzle.filter(part, checkSet, true);
					}
				}
			},
			"": function (checkSet, part, isXML) {
				var nodeCheck, doneName = done++,
					checkFn = dirCheck;
				if (typeof part === "string" && !rNonWord.test(part)) {
					part = part.toLowerCase();
					nodeCheck = part;
					checkFn = dirNodeCheck;
				}
				checkFn("parentNode", part, doneName, checkSet, nodeCheck, isXML);
			},
			"~": function (checkSet, part, isXML) {
				var nodeCheck, doneName = done++,
					checkFn = dirCheck;
				if (typeof part === "string" && !rNonWord.test(part)) {
					part = part.toLowerCase();
					nodeCheck = part;
					checkFn = dirNodeCheck;
				}
				checkFn("previousSibling", part, doneName, checkSet, nodeCheck, isXML);
			}
		},
		find: {
			ID: function (match, context, isXML) {
				if (typeof context.getElementById !== "undefined" && !isXML) {
					var m = context.getElementById(match[1]);
					return m && m.parentNode ? [m] : [];
				}
			},
			NAME: function (match, context) {
				if (typeof context.getElementsByName !== "undefined") {
					var ret = [],
						results = context.getElementsByName(match[1]);
					for (var i = 0, l = results.length; i < l; i++) {
						if (results[i].getAttribute("name") === match[1]) {
							ret.push(results[i]);
						}
					}
					return ret.length === 0 ? null : ret;
				}
			},
			TAG: function (match, context) {
				if (typeof context.getElementsByTagName !== "undefined") {
					return context.getElementsByTagName(match[1]);
				}
			}
		},
		preFilter: {
			CLASS: function (match, curLoop, inplace, result, not, isXML) {
				match = " " + match[1].replace(rBackslash, "") + " ";
				if (isXML) {
					return match;
				}
				for (var i = 0, elem;
				(elem = curLoop[i]) != null; i++) {
					if (elem) {
						if (not ^ (elem.className && (" " + elem.className + " ").replace(/[\t\n\r]/g, " ").indexOf(match) >= 0)) {
							if (!inplace) {
								result.push(elem);
							}
						} else if (inplace) {
							curLoop[i] = false;
						}
					}
				}
				return false;
			},
			ID: function (match) {
				return match[1].replace(rBackslash, "");
			},
			TAG: function (match, curLoop) {
				return match[1].replace(rBackslash, "").toLowerCase();
			},
			CHILD: function (match) {
				if (match[1] === "nth") {
					if (!match[2]) {
						Sizzle.error(match[0]);
					}
					match[2] = match[2].replace(/^\+|\s*/g, '');
					var test = /(-?)(\d*)(?:n([+\-]?\d*))?/.exec(match[2] === "even" && "2n" || match[2] === "odd" && "2n+1" || !/\D/.test(match[2]) && "0n+" + match[2] || match[2]);
					match[2] = (test[1] + (test[2] || 1)) - 0;
					match[3] = test[3] - 0;
				} else if (match[2]) {
					Sizzle.error(match[0]);
				}
				match[0] = done++;
				return match;
			},
			ATTR: function (match, curLoop, inplace, result, not, isXML) {
				var name = match[1] = match[1].replace(rBackslash, "");
				if (!isXML && Expr.attrMap[name]) {
					match[1] = Expr.attrMap[name];
				}
				match[4] = (match[4] || match[5] || "").replace(rBackslash, "");
				if (match[2] === "~=") {
					match[4] = " " + match[4] + " ";
				}
				return match;
			},
			PSEUDO: function (match, curLoop, inplace, result, not) {
				if (match[1] === "not") {
					if ((chunker.exec(match[3]) || "").length > 1 || /^\w/.test(match[3])) {
						match[3] = Sizzle(match[3], null, null, curLoop);
					} else {
						var ret = Sizzle.filter(match[3], curLoop, inplace, true ^ not);
						if (!inplace) {
							result.push.apply(result, ret);
						}
						return false;
					}
				} else if (Expr.match.POS.test(match[0]) || Expr.match.CHILD.test(match[0])) {
					return true;
				}
				return match;
			},
			POS: function (match) {
				match.unshift(true);
				return match;
			}
		},
		filters: {
			enabled: function (elem) {
				return elem.disabled === false && elem.type !== "hidden";
			},
			disabled: function (elem) {
				return elem.disabled === true;
			},
			checked: function (elem) {
				return elem.checked === true;
			},
			selected: function (elem) {
				if (elem.parentNode) {
					elem.parentNode.selectedIndex;
				}
				return elem.selected === true;
			},
			parent: function (elem) {
				return !!elem.firstChild;
			},
			empty: function (elem) {
				return !elem.firstChild;
			},
			has: function (elem, i, match) {
				return !!Sizzle(match[3], elem).length;
			},
			header: function (elem) {
				return (/h\d/i).test(elem.nodeName);
			},
			text: function (elem) {
				return "text" === elem.getAttribute('type');
			},
			radio: function (elem) {
				return "radio" === elem.type;
			},
			checkbox: function (elem) {
				return "checkbox" === elem.type;
			},
			file: function (elem) {
				return "file" === elem.type;
			},
			password: function (elem) {
				return "password" === elem.type;
			},
			submit: function (elem) {
				return "submit" === elem.type;
			},
			image: function (elem) {
				return "image" === elem.type;
			},
			reset: function (elem) {
				return "reset" === elem.type;
			},
			button: function (elem) {
				return "button" === elem.type || elem.nodeName.toLowerCase() === "button";
			},
			input: function (elem) {
				return (/input|select|textarea|button/i).test(elem.nodeName);
			}
		},
		setFilters: {
			first: function (elem, i) {
				return i === 0;
			},
			last: function (elem, i, match, array) {
				return i === array.length - 1;
			},
			even: function (elem, i) {
				return i % 2 === 0;
			},
			odd: function (elem, i) {
				return i % 2 === 1;
			},
			lt: function (elem, i, match) {
				return i < match[3] - 0;
			},
			gt: function (elem, i, match) {
				return i > match[3] - 0;
			},
			nth: function (elem, i, match) {
				return match[3] - 0 === i;
			},
			eq: function (elem, i, match) {
				return match[3] - 0 === i;
			}
		},
		filter: {
			PSEUDO: function (elem, match, i, array) {
				var name = match[1],
					filter = Expr.filters[name];
				if (filter) {
					return filter(elem, i, match, array);
				} else if (name === "contains") {
					return (elem.textContent || elem.innerText || Sizzle.getText([elem]) || "").indexOf(match[3]) >= 0;
				} else if (name === "not") {
					var not = match[3];
					for (var j = 0, l = not.length; j < l; j++) {
						if (not[j] === elem) {
							return false;
						}
					}
					return true;
				} else {
					Sizzle.error(name);
				}
			},
			CHILD: function (elem, match) {
				var type = match[1],
					node = elem;
				switch (type) {
				case "only":
				case "first":
					while ((node = node.previousSibling)) {
						if (node.nodeType === 1) {
							return false;
						}
					}
					if (type === "first") {
						return true;
					}
					node = elem;
				case "last":
					while ((node = node.nextSibling)) {
						if (node.nodeType === 1) {
							return false;
						}
					}
					return true;
				case "nth":
					var first = match[2],
						last = match[3];
					if (first === 1 && last === 0) {
						return true;
					}
					var doneName = match[0],
						parent = elem.parentNode;
					if (parent && (parent.sizcache !== doneName || !elem.nodeIndex)) {
						var count = 0;
						for (node = parent.firstChild; node; node = node.nextSibling) {
							if (node.nodeType === 1) {
								node.nodeIndex = ++count;
							}
						}
						parent.sizcache = doneName;
					}
					var diff = elem.nodeIndex - last;
					if (first === 0) {
						return diff === 0;
					} else {
						return (diff % first === 0 && diff / first >= 0);
					}
				}
			},
			ID: function (elem, match) {
				return elem.nodeType === 1 && elem.getAttribute("id") === match;
			},
			TAG: function (elem, match) {
				return (match === "*" && elem.nodeType === 1) || elem.nodeName.toLowerCase() === match;
			},
			CLASS: function (elem, match) {
				return (" " + (elem.className || elem.getAttribute("class")) + " ").indexOf(match) > -1;
			},
			ATTR: function (elem, match) {
				var name = match[1],
					result = Expr.attrHandle[name] ? Expr.attrHandle[name](elem) : elem[name] != null ? elem[name] : elem.getAttribute(name),
					value = result + "",
					type = match[2],
					check = match[4];
				return result == null ? type === "!=" : type === "=" ? value === check : type === "*=" ? value.indexOf(check) >= 0 : type === "~=" ? (" " + value + " ").indexOf(check) >= 0 : !check ? value && result !== false : type === "!=" ? value !== check : type === "^=" ? value.indexOf(check) === 0 : type === "$=" ? value.substr(value.length - check.length) === check : type === "|=" ? value === check || value.substr(0, check.length + 1) === check + "-" : false;
			},
			POS: function (elem, match, i, array) {
				var name = match[2],
					filter = Expr.setFilters[name];
				if (filter) {
					return filter(elem, i, match, array);
				}
			}
		}
	};
	var origPOS = Expr.match.POS,
		fescape = function (all, num) {
			return "\\" + (num - 0 + 1);
		};
	for (var type in Expr.match) {
		Expr.match[type] = new RegExp(Expr.match[type].source + (/(?![^\[]*\])(?![^\(]*\))/.source));
		Expr.leftMatch[type] = new RegExp(/(^(?:.|\r|\n)*?)/.source + Expr.match[type].source.replace(/\\(\d+)/g, fescape));
	}
	var makeArray = function (array, results) {
			array = ve.lang.arg2Arr(array, 0);
			if (results) {
				results.push.apply(results, array);
				return results;
			}
			return array;
		};
	try {
		Array.prototype.slice.call(document.documentElement.childNodes, 0)[0].nodeType;
	} catch (e) {
		makeArray = function (array, results) {
			var i = 0,
				ret = results || [];
			if (toString.call(array) === "[object Array]") {
				Array.prototype.push.apply(ret, array);
			} else {
				if (typeof array.length === "number") {
					for (var l = array.length; i < l; i++) {
						ret.push(array[i]);
					}
				} else {
					for (; array[i]; i++) {
						ret.push(array[i]);
					}
				}
			}
			return ret;
		};
	}
	var sortOrder, siblingCheck;
	if (document.documentElement.compareDocumentPosition) {
		sortOrder = function (a, b) {
			if (a === b) {
				hasDuplicate = true;
				return 0;
			}
			if (!a.compareDocumentPosition || !b.compareDocumentPosition) {
				return a.compareDocumentPosition ? -1 : 1;
			}
			return a.compareDocumentPosition(b) & 4 ? -1 : 1;
		};
	} else {
		sortOrder = function (a, b) {
			var al, bl, ap = [],
				bp = [],
				aup = a.parentNode,
				bup = b.parentNode,
				cur = aup;
			if (a === b) {
				hasDuplicate = true;
				return 0;
			} else if (aup === bup) {
				return siblingCheck(a, b);
			} else if (!aup) {
				return -1;
			} else if (!bup) {
				return 1;
			}
			while (cur) {
				ap.unshift(cur);
				cur = cur.parentNode;
			}
			cur = bup;
			while (cur) {
				bp.unshift(cur);
				cur = cur.parentNode;
			}
			al = ap.length;
			bl = bp.length;
			for (var i = 0; i < al && i < bl; i++) {
				if (ap[i] !== bp[i]) {
					return siblingCheck(ap[i], bp[i]);
				}
			}
			return i === al ? siblingCheck(a, bp[i], -1) : siblingCheck(ap[i], b, 1);
		};
		siblingCheck = function (a, b, ret) {
			if (a === b) {
				return ret;
			}
			var cur = a.nextSibling;
			while (cur) {
				if (cur === b) {
					return -1;
				}
				cur = cur.nextSibling;
			}
			return 1;
		};
	}
	Sizzle.getText = function (elems) {
		var ret = "",
			elem;
		for (var i = 0; elems[i]; i++) {
			elem = elems[i];
			if (elem.nodeType === 3 || elem.nodeType === 4) {
				ret += elem.nodeValue;
			} else if (elem.nodeType !== 8) {
				ret += Sizzle.getText(elem.childNodes);
			}
		}
		return ret;
	};
	(function () {
		var form = document.createElement("div"),
			id = "script" + (new Date()).getTime(),
			root = document.documentElement;
		form.innerHTML = "<a name='" + id + "'/>";
		root.insertBefore(form, root.firstChild);
		if (document.getElementById(id)) {
			Expr.find.ID = function (match, context, isXML) {
				if (typeof context.getElementById !== "undefined" && !isXML) {
					var m = context.getElementById(match[1]);
					return m ? m.id === match[1] || typeof m.getAttributeNode !== "undefined" && m.getAttributeNode("id").nodeValue === match[1] ? [m] : undefined : [];
				}
			};
			Expr.filter.ID = function (elem, match) {
				var node = typeof elem.getAttributeNode !== "undefined" && elem.getAttributeNode("id");
				return elem.nodeType === 1 && node && node.nodeValue === match;
			};
		}
		root.removeChild(form);
		root = form = null;
	})();
	(function () {
		var div = document.createElement("div");
		div.appendChild(document.createComment(""));
		if (div.getElementsByTagName("*").length > 0) {
			Expr.find.TAG = function (match, context) {
				var results = context.getElementsByTagName(match[1]);
				if (match[1] === "*") {
					var tmp = [];
					for (var i = 0; results[i]; i++) {
						if (results[i].nodeType === 1) {
							tmp.push(results[i]);
						}
					}
					results = tmp;
				}
				return results;
			};
		}
		div.innerHTML = "<a href='#'></a>";
		if (div.firstChild && typeof div.firstChild.getAttribute !== "undefined" && div.firstChild.getAttribute("href") !== "#") {
			Expr.attrHandle.href = function (elem) {
				return elem.getAttribute("href", 2);
			};
		}
		div = null;
	})();
	if (document.querySelectorAll) {
		(function () {
			var oldSizzle = Sizzle,
				id = "__sizzle__";
			Sizzle = function (query, context, extra, seed) {
				context = context || document;
				if (!seed && !Sizzle.isXML(context)) {
					var match = rSpeedUp.exec(query);
					if (match && (context.nodeType === 1 || context.nodeType === 9)) {
						if (match[1]) {
							return makeArray(context.getElementsByTagName(query), extra);
						} else if (match[2] && Expr.find.CLASS && context.getElementsByClassName) {
							return makeArray(context.getElementsByClassName(match[2]), extra);
						}
					}
					if (context.nodeType === 9) {
						if (query === "body" && context.body) {
							return makeArray([context.body], extra);
						} else if (match && match[3]) {
							var elem = context.getElementById(match[3]);
							if (elem && elem.parentNode) {
								if (elem.id === match[3]) {
									return makeArray([elem], extra);
								}
							} else {
								return makeArray([], extra);
							}
						}
						try {
							return makeArray(context.querySelectorAll(query), extra);
						} catch (qsaError) {}
					} else if (context.nodeType === 1 && context.nodeName.toLowerCase() !== "object") {
						var oldContext = context,
							old = context.getAttribute("id"),
							nid = old || id,
							hasParent = context.parentNode,
							relativeHierarchySelector = /^\s*[+~]/.test(query);
						if (!old) {
							context.setAttribute("id", nid);
						} else {
							nid = nid.replace(/'/g, "\\$&");
						}
						if (relativeHierarchySelector && hasParent) {
							context = context.parentNode;
						}
						try {
							if (!relativeHierarchySelector || hasParent) {
								return makeArray(context.querySelectorAll("[id='" + nid + "'] " + query), extra);
							}
						} catch (pseudoError) {} finally {
							if (!old) {
								oldContext.removeAttribute("id");
							}
						}
					}
				}
				return oldSizzle(query, context, extra, seed);
			};
			for (var prop in oldSizzle) {
				Sizzle[prop] = oldSizzle[prop];
			}
		})();
	}
	(function () {
		var html = document.documentElement,
			matches = html.matchesSelector || html.mozMatchesSelector || html.webkitMatchesSelector || html.msMatchesSelector,
			pseudoWorks = false;
		try {
			matches.call(document.documentElement, "[test!='']:sizzle");
		} catch (pseudoError) {
			pseudoWorks = true;
		}
		if (matches) {
			Sizzle.matchesSelector = function (node, expr) {
				expr = expr.replace(/\=\s*([^'"\]]*)\s*\]/g, "='$1']");
				if (!Sizzle.isXML(node)) {
					try {
						if (pseudoWorks || !Expr.match.PSEUDO.test(expr) && !/!=/.test(expr)) {
							return matches.call(node, expr);
						}
					} catch (e) {}
				}
				return Sizzle(expr, null, null, [node]).length > 0;
			};
		}
	})();
	Expr.order.splice(1, 0, "CLASS");
	Expr.find.CLASS = function (match, context, isXML) {
		if (typeof context.getElementsByClassName !== "undefined" && !isXML) {
			return context.getElementsByClassName(match[1]);
		}
	};

	function dirNodeCheck(dir, cur, doneName, checkSet, nodeCheck, isXML) {
		for (var i = 0, l = checkSet.length; i < l; i++) {
			var elem = checkSet[i];
			if (elem) {
				var match = false;
				elem = elem[dir];
				while (elem) {
					if (elem.sizcache === doneName) {
						match = checkSet[elem.sizset];
						break;
					}
					if (elem.nodeType === 1 && !isXML) {
						elem.sizcache = doneName;
						elem.sizset = i;
					}
					if (elem.nodeName.toLowerCase() === cur) {
						match = elem;
						break;
					}
					elem = elem[dir];
				}
				checkSet[i] = match;
			}
		}
	}

	function dirCheck(dir, cur, doneName, checkSet, nodeCheck, isXML) {
		for (var i = 0, l = checkSet.length; i < l; i++) {
			var elem = checkSet[i];
			if (elem) {
				var match = false;
				elem = elem[dir];
				while (elem) {
					if (elem.sizcache === doneName) {
						match = checkSet[elem.sizset];
						break;
					}
					if (elem.nodeType === 1) {
						if (!isXML) {
							elem.sizcache = doneName;
							elem.sizset = i;
						}
						if (typeof cur !== "string") {
							if (elem === cur) {
								match = true;
								break;
							}
						} else if (Sizzle.filter(cur, [elem]).length > 0) {
							match = elem;
							break;
						}
					}
					elem = elem[dir];
				}
				checkSet[i] = match;
			}
		}
	}
	if (document.documentElement.compareDocumentPosition) {
		Sizzle.contains = function (a, b) {
			return !!(a.compareDocumentPosition(b) & 16);
		};
	} else if (document.documentElement.contains) {
		Sizzle.contains = function (a, b) {
			if (a !== b && a.contains && b.contains) {
				return a.contains(b);
			} else if (!b || b.nodeType == 9) {
				return false;
			} else if (b === a) {
				return true;
			} else {
				return Sizzle.contains(a, b.parentNode);
			}
		};
	} else {
		Sizzle.contains = function () {
			return false;
		};
	}
	Sizzle.isXML = function (elem) {
		var documentElement = (elem ? elem.ownerDocument || elem : 0).documentElement;
		return documentElement ? documentElement.nodeName !== "HTML" : false;
	};
	var posProcess = function (selector, context) {
		var match, tmpSet = [],
			later = "",
			root = context.nodeType ? [context] : context;
		while ((match = Expr.match.PSEUDO.exec(selector))) {
			later += match[0];
			selector = selector.replace(Expr.match.PSEUDO, "");
		}
		selector = Expr.relative[selector] ? selector + "*" : selector;
		for (var i = 0, l = root.length; i < l; i++) {
			Sizzle(selector, root[i], tmpSet);
		}
		return Sizzle.filter(later, tmpSet);
	};
	dom.selector = dom.find = Sizzle;
	dom.one = function(){
		var result = dom.find.apply(dom.find, arguments);
		if(result && result.length){
			return result[0];
		}
		return null;
	};
})(VEditor, VEditor.dom);