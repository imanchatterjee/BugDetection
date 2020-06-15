/*******************************************************************************
 * @license
 * Copyright (c) 2011, 2012 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials are made 
 * available under the terms of the Eclipse Public License v1.0 
 * (http://www.eclipse.org/legal/epl-v10.html), and the Eclipse Distribution 
 * License v1.0 (http://www.eclipse.org/org/documents/edl-v10.html). 
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
/*eslint-env browser, amd*/
define(['orion/URITemplate'], function(URITemplate) {

var orion = orion || {};

/**
 * Utility methods
 * @namespace orion.compareUtils 
 */
 
orion.compareUtils = orion.compareUtils || {};

/**
 * Look up the mapper item by a given line index (zero based)
 * @static
 * @param {Array} mapper , the mapper generated by the  diffParser
 * @param {int} action , mapperColumnIndex , the column index of the mapper , should be 0 or 1
 * @param {int} lineIndex , the given line index
 * @return {Object} the object with two fields : the mapper index that collides with the lineIndex , the start line of the mapper
 */
orion.compareUtils.lookUpMapper = function(mapper , mapperColumnIndex , lineIndex) {
	var curLineindex = 0;//zero based
	for (var i = 0 ; i < mapper.length ; i++){
		var size = mapper[i][mapperColumnIndex];
		if(size === 0){
			size = 1;
		}
		if(lineIndex >= curLineindex && lineIndex < (curLineindex + size)){
			return {mapperIndex:i , startFrom:curLineindex};
		}
		if(i === (mapper.length - 1 )){
			break;
		}
		curLineindex += mapper[i][mapperColumnIndex];
	}
	return  {mapperIndex:mapper.length-1 , startFrom:curLineindex};
};

orion.compareUtils.lookUpLineIndex = function(mapper , mapperColumnIndex , mapperIndex){
	if(mapperIndex === 0){
		return 0;
	}
	var curLineindex = 0;//zero based
	for (var i = 0 ; i < mapperIndex ; i++){
		curLineindex += mapper[i][mapperColumnIndex];
	}
	return curLineindex;
};

orion.compareUtils.updateMapper = function(mapper , mapperColumnIndex , startLineIndex, removedLineCount, addedLineCount){
	if(removedLineCount === addedLineCount){
		return;
	}
	if(removedLineCount > 0 || addedLineCount > 0){
		var mapperItem = orion.compareUtils.lookUpMapper(mapper , mapperColumnIndex ,startLineIndex);
		if(mapperItem.mapperIndex < 0){
			return;
		}
		if(removedLineCount > 0){
			var linesLeft = removedLineCount;
			var startInMapper = startLineIndex - mapperItem.startFrom;
			for(var i = mapperItem.mapperIndex ; i < mapper.length ; i++){
				var wipeOutLines = mapper[i][mapperColumnIndex] - startInMapper;
				if(linesLeft <= wipeOutLines){
					mapper[i][mapperColumnIndex] -= linesLeft;
					break;
				}
				mapper[i][mapperColumnIndex] -= wipeOutLines;
				linesLeft -= wipeOutLines;
				startInMapper = 0;
			}
		}
		if(addedLineCount > 0){
			mapper[mapperItem.mapperIndex][mapperColumnIndex] += addedLineCount;
		}
	}
};

orion.compareUtils.overlapMapper = function(mapperItem , mapperColumnIndex , startLineIndex, lineFrom , lineTo){
	var endLineIndex = startLineIndex + mapperItem[mapperColumnIndex] - 1;
	
	if(endLineIndex < startLineIndex){
		endLineIndex = startLineIndex;
	}
	if(lineTo < lineFrom){
		lineTo = lineFrom;
	}
	if (endLineIndex < lineFrom || lineTo < startLineIndex){
		return false;
	}
	return true; 
};

orion.compareUtils.findFirstDiff = function(mapper , mapperColumnIndex , lineFrom , lineTo){
	var curLineIndex = 0;
	var retValue = null;
	for (var i = 0 ; i < mapper.length ; i++){
		if(curLineIndex > lineTo){
			break;
		}
		if(orion.compareUtils.overlapMapper( mapper[i] , mapperColumnIndex , curLineIndex , lineFrom , lineTo)){
			retValue = {mapperIndex:i , startFrom:curLineIndex };
			if( mapper[i][2] !== 0 ){
				break;
			}
		}
		curLineIndex  +=  mapper[i][mapperColumnIndex];
	}
	return  retValue;
};


//returns the line index at the top of the other editor , when scroll happens on the eidotr
orion.compareUtils.matchMapper = function(mapper , mapperColumnIndex , lineFrom , lineTo){
	if(mapper.length === 0){
		return lineFrom;
	}
	var baseLine = lineFrom + Math.round((lineTo -lineFrom)/3);
	var first = orion.compareUtils.findFirstDiff(mapper , mapperColumnIndex , lineFrom , lineTo);
	var mapperEndAt = mapper[first.mapperIndex][mapperColumnIndex] === 0 ? first.startFrom : first.startFrom + mapper[first.mapperIndex][mapperColumnIndex] -1;
	
	var startLineAtOther = orion.compareUtils.lookUpLineIndex(mapper , 1-mapperColumnIndex , first.mapperIndex);
	var delta = first.startFrom - lineFrom;
	
	if( mapper[first.mapperIndex][2] === 0){
		return (startLineAtOther -delta);
	}
	if(baseLine >= first.startFrom && baseLine <= mapperEndAt){
		return startLineAtOther -  Math.round((lineTo -lineFrom)/3);
	}	
	if(baseLine < first.startFrom){
		return (startLineAtOther -delta);
	}
	
	var mapperEndAtOther = mapper[first.mapperIndex][1-mapperColumnIndex] === 0 ? startLineAtOther : startLineAtOther + mapper[first.mapperIndex][1-mapperColumnIndex] -1;
	return ( mapperEndAtOther- (mapperEndAt - lineFrom));
	
};

orion.compareUtils.getMapperLineCount = function(mapper){
	var curLineindex = 0;//zero based
	for (var i = 0 ; i < mapper.length ; i++){
		curLineindex += Math.max(mapper[i][0] ,mapper[i][1]);
	}
	return curLineindex;
};

orion.compareUtils.getAnnotationIndex = function(annotations, lineIndex){
	for (var i = 0 ; i < annotations.length ; i++){
		if(annotations[i][0] === lineIndex){
			return i;
		}
	}
	return -1;
};

orion.compareUtils.getAnnotationMapperIndex = function(annotations, annotationIndex){
	return annotations[annotationIndex][1];
};

orion.compareUtils.getAnnotationIndexByMapper = function(annotations, mapperIndex){
	for (var i = 0 ; i < annotations.length ; i++){
		if(annotations[i][1] === mapperIndex){
			return {current:i,prev:i-1,next:i+1};
		}
	}
	return {current:-1,prev:-1,next:-1};
};

orion.compareUtils.isMapperConflict = function(mapper, mapperIndex){
	if(mapperIndex < 0){
		return false;
	}
	if(!mapper[mapperIndex][3]){
		return false;
	}
	return mapper[mapperIndex][3] === 1;
};

orion.compareUtils.mergeDiffBlocks = function(oldTextModel, newDiffBlocks, mapper, diffArray, diffArraySubstrIndex, lineDelim){
	for(var i = 0; i < newDiffBlocks.length; i++){
		var startLineIndex = newDiffBlocks[i][0];
		var mapperIndex = newDiffBlocks[i][1];
		var mapperItem = mapper[mapperIndex];
		if(mapperItem[0] > 0){
			var text = "";
			for(var j = 0; j < mapperItem[0]; j++){
				var lineText = diffArray[mapperItem[2]-1+j];
				if(lineText) {
					text = text + lineText.substring(diffArraySubstrIndex) + lineDelim;
				} else {
					text = text + lineDelim;
				}
			}
			var lineCount = oldTextModel.getLineCount();
			if(startLineIndex >= lineCount){
				//Check if the last line of the text model has line delimeter or not
				var lineEnd1 = oldTextModel.getLineEnd(lineCount - 1);
				var lineEnd2 = oldTextModel.getLineEnd(lineCount - 1, true);
				if(lineEnd1 === lineEnd2){//last line has no line delimeter, we need to fake one
					oldTextModel.setText(lineDelim, lineEnd1, lineEnd1);
					startLineIndex = lineCount;
				} else {
					startLineIndex = lineCount -1;
				}
			}
			var startOffset = oldTextModel.getLineStart(startLineIndex);
			oldTextModel.setText(text, startOffset, startOffset);
		}
	}
};

orion.compareUtils.convertMergedLineNumber = function(mapper, lineindex/*zero based*/){
	var mapperItem = orion.compareUtils.lookUpMapper(mapper, 0, lineindex);
	var returnValue = lineindex;
	if(mapperItem.mapperIndex < 0){
		return returnValue;
	}
	for(var i = 0; i < mapperItem.mapperIndex; i++){
		if(mapper[i][2] !== 0) {
			returnValue = returnValue + mapper[i][0];
		}
	}
	return returnValue;
};

orion.compareUtils.generateCompareHref = function(diffLocation, options) {
	var base =  require.toUrl("compare/compare.html"); //$NON-NLS-0$
	if(!options.conflict){
		options.conflict = undefined;
	}
	if(!options.readonly){
		options.readonly = undefined;
	}
	var href = new URITemplate(base + "#{,resource,params*}").expand({ //$NON-NLS-0$
		resource: diffLocation,
		params: options
	});
	return href;
};

orion.compareUtils.generateCompareTreeHref = function(folderToCompare, options) {
	var base =  require.toUrl("compare-tree/compare-tree.html"); //$NON-NLS-0$
	var href = new URITemplate(base + "#{,resource,params*}").expand({ //$NON-NLS-0$
		resource: folderToCompare,
		params: options
	});
	return href;
};

return orion.compareUtils;
});