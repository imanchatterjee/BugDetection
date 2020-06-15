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

CV = namedtuple('CV', ['consequent_name', 'type'])

class LearningData(object):
    def __init__(self):
        self.file_to_conditionalStatement = dict() # string to set of return statements
        self.stats = {}

    def resetStats(self):
        self.stats = {}

    def pre_scan(self, training_data_paths, validation_data_paths):
        all_conditionalStatement_set = set()
        for condition_val in Util.DataReader(training_data_paths):
            file = condition_val["src"].split(" : ")[0]
            condition_arguments = self.file_to_conditionalStatement.setdefault(file, set())
            condition_arguments.add(CV(condition_val["consequent_name"], condition_val["consequent_type"]))
            
        for condition_val in Util.DataReader(validation_data_paths):
            file = condition_val["src"].split(" : ")[0]
            condition_arguments = self.file_to_conditionalStatement.setdefault(file, set())
            condition_arguments.add(CV(condition_val["consequent_name"], condition_val["consequent_type"]))
            
        #self.all_operators = list(all_returnStatment_set)

  
    def code_to_xy_pairs(self, conditional_val, xs, ys, name_to_vector, type_to_vector, node_type_to_vector, code_pieces):
        #Add Function name if want to
        cons_argument = conditional_val["consequent_name"]
        cons_argument_type = conditional_val["consequent_type"]
        parent = conditional_val["parent"]
        grand_parent = conditional_val["grandParent"]
        src = conditional_val["src"]
        if not (cons_argument in name_to_vector):
            return
        
        cons_argument_vector = name_to_vector[cons_argument]
        cons_type_vector = type_to_vector.get(cons_argument_type, [0]*type_embedding_size)
        try:
          parent_vector = node_type_to_vector[parent]
        
          
          grand_parent_vector = node_type_to_vector[grand_parent]
        except:
          return
        
        # find an alternative operand in the same file
        file = src.split(" : ")[0]
       
        
        # for all xy-pairs: y value = probability that incorrect
        x_correct = cons_argument_vector + cons_type_vector + parent_vector + grand_parent_vector
        y_correct = [0]
        xs.append(x_correct)
        ys.append(y_correct)
        code_pieces.append(CodePiece(cons_argument, cons_argument_type, src))
        
        #LIT:null
        namestring=['ID:undefined','LIT:null']
        typestring=['undefined','null']
        other_operand_vector = name_to_vector[random.choice(namestring)]
        other_operand_type_vector = type_to_vector[random.choice(typestring)]
     
        # replace if statement with the alternative one
        x_incorrect = cons_argument_vector +  other_operand_type_vector +  parent_vector + grand_parent_vector
        #x_incorrect = other_operand_vector +  other_operand_type_vector +  parent_vector + grand_parent_vector
        y_incorrect = [1]
        xs.append(x_incorrect)
        ys.append(y_incorrect)
        code_pieces.append(CodePiece(cons_argument, cons_argument_type, src))
        
    def anomaly_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_orig
    
    def normal_score(self, y_prediction_orig, y_prediction_changed):
        return y_prediction_changed
                   
