(function() {

    const fs = require("fs");
    const estraverse = require("estraverse");
    const util = require("./jsExtractionUtil");

    function visitCode(ast, locationMap, path, all_return_value, fileIDStr) {
        console.log("Reading " + path);

  

        const parentStack = [];
        const if_statement = [];
        let tokenID = 1;
        estraverse.traverse(ast, {
            enter:function(node, parent) {
                if (parent) parentStack.push(parent);
                if (node.type === "ConditionalExpression") {
                    let functionName = util.getNameOfFunction(node, parent);
                    let conseqType = util.getTypeOfASTNode(node.consequent);
                    let conseqName = util.getNameOfASTNode(node.consequent);
                    //return type of value that is right to the 
                    //if(node.right  )
                    
                    const parentName = parent.type;
                    const grandParentName = parentStack.length > 1 ? parentStack[parentStack.length - 2].type : "";
//                     console.log(parentName,conseqType,conseqName);
	             //if (typeof conseqName !== 'string' ){
                     if (parentName ){

			 //console.log(typeof conseqName);
		//	return;
                      //if(typeof conseqName !== "undefined" ) {
                     // if(functionName)  {
                      let locString = path + " : " + node.loc.start.line + " - " + node.loc.end.line;
                      if(!conseqName){
                         conseqName='blank';  
                                }
                       //console.log(conseqName);
                      const if_stat = {
                            function_name:functionName,
                            consequent_type:conseqType,
                            consequent_name:conseqName,
                            parent:parentName,
                            grandParent:grandParentName,
                            src:locString
                        };
                        if_statement.push(if_stat);
                    
                        tokenID += 1;
                    }
                }
            },
            leave:function(node, parent) {
                if (parent) parentStack.pop();
            }
        });
        all_return_value.push(...if_statement);
        console.log("Added if statement. Total now: " + all_return_value.length);
    }

    module.exports.visitCode = visitCode;

})();
