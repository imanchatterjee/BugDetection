import Util
from collections import Counter
from collections import namedtuple
import random

type_embedding_size = 5
node_type_embedding_size = 8

class CodePiece(object):
    def __init__(self, return_name, return_type, src):
        self.return_name = return_name
        self.return_type = return_type
        self.src = src
 
    def to_message(self):
        return str(self.src) + " | " + str(self.return_name) + " | " + str(self.return_type) 

RV = namedtuple('RV', ['return_name', 'type'])

class LearningData(object):
    def __init__(self):
        self.file_to_returnStatment = dict() # string to set of return statements
        self.stats = {}

    def resetStats(self):
        self.stats = {}

    def pre_scan(self, training_data_paths, validation_data_paths):
        all_returnStatment_set = set()
        for ret_val in Util.DataReader(training_data_paths):
            file = ret_val["src"].split(" : ")[0]
            return_arguments = self.file_to_returnStatment.setdefault(file, set())
            return_arguments.add(RV(ret_val["return_name"], ret_val["return_type"]))
            
        for ret_val in Util.DataReader(validation_data_paths):
            file = ret_val["src"].split(" : ")[0]
            return_arguments = self.file_to_returnStatment.setdefault(file, set())
            return_arguments.add(RV(ret_val["return_name"], ret_val["return_type"]))
            
        #self.all_operators = list(all_returnStatment_set)

  
    def code_to_xy_pairs(self, ret_val, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, code_pieces):
        #Add Function name if want to
        return_argument = ret_val["return_name"]
        return_argument_type = ret_val["return_type"]
        parent = ret_val["parent"]
        grand_parent = ret_val["grandParent"]
        src = ret_val["src"]
        if not (return_argument in name_to_vector):
            return
        
        return_argument_vector = name_to_vector[return_argument]
        return_type_vector = type_to_vector.get(return_argument_type, [0]*type_embedding_size)
        parent_vector = node_type_to_vector[parent]
        try:
            grand_parent_vector = node_type_to_vector[grand_parent]
        except:
            return
        # find an alternative operand in the same file
        file = src.split(" : ")[0]
        all_operands = self.file_to_returnStatment[file]
        tries_left = 100
        found = False
        while (not found) and tries_left > 0:
            other_operand = random.choice(list(all_operands))
            if other_operand.return_name in name_to_vector: #and other_operand.return_name != to_replace_operand:
                found = True
            tries_left -= 1
            
        if not found:
            return
        
        # for all xy-pairs: y value = probability that incorrect
        x_correct = return_argument_vector + return_type_vector + parent_vector + grand_parent_vector
        y_correct = [0]
        xs.append(x_correct)
        ys.append(y_correct)
        code_pieces.append(CodePiece(return_argument, return_argument_type, src))
        
        #print("Other Opernad examples: ", other_operand)
        other_operand_vector = name_to_vector[other_operand.return_name]
        other_operand_type_vector = type_to_vector[other_operand.type]
        # replace return statmetn with the alternative one
        x_incorrect = other_operand_vector +  other_operand_type_vector +  parent_vector + grand_parent_vector
        #print("x_incorrect: ", x_incorrect)
        y_incorrect = [1]
        xs.append(x_incorrect)
        ys.append(y_incorrect)
        code_pieces.append(CodePiece(return_argument, return_argument_type, src))
        
    def anomaly_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_orig - y_prediction_changed
    
    def normal_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_changed - y_prediction_orig
                   