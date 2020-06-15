(function() {

    const fs = require("fs");
    const estraverse = require("estraverse");
    const util = require("./jsExtractionUtil");

    function visitCode(ast, locationMap, path, all_return_value, fileIDStr) {
        console.log("Reading " + path);

  

        const parentStack = [];
        const return_statement = [];
        let tokenID = 1;
        estraverse.traverse(ast, {
            enter:function(node, parent) {
                if (parent) parentStack.push(parent);
                if (node.type === "ReturnStatement") {
                    let functionName = util.getNameOfFunction(node, parent);
                    //return type of value that is right to the 
                    //if(node.right  )
                    let returnType = util.getTypeOfASTNode(node.argument);
                    let returnName = util.getNameOfASTNode(node.argument);
                    const parentName = parent.type;
                    const grandParentName = parentStack.length > 1 ? parentStack[parentStack.length - 2].type : "";
                    if (typeof returnName !== "undefined" ) {
                        let locString = path + " : " + node.loc.start.line + " - " + node.loc.end.line;
                        const return_stat = {
                            function_name:functionName,
                            return_type:returnType,
                            return_name:returnName,
                            parent:parentName,
                            grandParent:grandParentName,
                            src:locString
                        };
                        return_statement.push(return_stat);
                    
                        tokenID += 1;
                    }
                }
            },
            leave:function(node, parent) {
                if (parent) parentStack.pop();
            }
        });
        all_return_value.push(...return_statement);
        console.log("Added return statement. Total now: " + all_return_value.length);
    }

    module.exports.visitCode = visitCode;

})();
