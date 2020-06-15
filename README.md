** DeepBugs **
 
Automatic Bug detection is a critical strategy that helps saves a lot of the developer’s time. A few popular ways to perform bug detection are genetic programming, machine learning and pattern matching. DeepBugs paper uses machine learning path to train three model where each can predict a specific kind of bug. In this paper, we extend the DeepBugs to train two more models. These models will find Wrong Return Statement bug and Empty BlockStatement bug. We create a training data from the given corpus with new bugs. Models trained on newly created data, are capableof predicting the same bug in an unseen code. The new models willbe trained on a corpus of 150,000 JavaScript files

Artifacts:
The programs that we need to train a model on are saved in data folder. We have a small corpus and a large corpus of programs to run Deep Bugs on.  There are two parts to the DeepBugs algorithm: First parses the training data set from given code examples. Second, Introduce a bug in correct files to create training data, train a model on newly created data and predict bugs on unseen code. Parsing of data from given program is done in javascript. All the programs related to extraction are saved in a subfolder named javascript inside src. Training of machine learning model is doen in python. All the programs related to training and prediction are saved in subfolder python inside src. 

Required Languages and Packages:
To run DeepBugs you should have Node.js and Python3 downloaded. In Node.js, required npm modules are acorn, estraverse, walk-sync. For python, keras, scipy,numpy, sklearn, TensorFlow

We would run a total of two commands, one for parsing the code and other for training and testing. All Commands should be run inside chatterjee-kaur folder.

Step 1.The first step is to parse the javascript files (data code) into training examples. To do so, at first we need to select what bug we want to implement and what data we want to use. There are three main bugs that DeepBugs already trained the model on. We have added two more: Incorrect Return Statement and Empty Block Statement. Below command extracts only relevant information of Incorrect Return Statement from 50 training Examples. Note that there is a big corpus available to train the model, but that would take significantly more time. The command is: 

node src/javascript/extractFromJS.js return_statement --parallel 4 data/js/programs_50_training.txt data/js/programs_50

To evaluate the data, run the same command again, but by replacing training.txt to eval.txt. The command should produce a jsfile with name return_statement_*.js file. These files would be saved in chatter-kaur/ folder. Program uses timestamp at place of *. Rename the produced files for easy use since they will be used in second step. Command will also print information about the number of examples it is extracting and parsing. For Empty Block Statement bug, replace 𝑟𝑒𝑡𝑢𝑟𝑛_𝑠𝑡𝑎𝑡𝑒𝑚𝑒𝑛𝑡 with 𝑖𝑓_𝑠𝑡𝑎𝑡𝑒𝑚𝑒𝑛𝑡 in the command. For other bugs such as Swapped Arguments, use 𝑐𝑎𝑙𝑙𝑠 and for wrong binaryoperator, use 𝑏𝑖𝑛𝑂𝑝𝑠. Now we have programs parsed into desired format. 

Step 2. Next step is to implement bug to create complete training data set and train the model based upon that. The following command is used for this purpose.

python3 src/python/BugDetection.py IncorrectReturnStatement --learn src/token_to_vector.json src/type_to_vector.json src/node_type_to_vector.json --trainingData return_statement_*.json --validationData return_statement_*.json

Modify the statement by replacing return_statement_*.json withrenamed files. First occurrence refers to training file and the second refers to evaluation file. Output of the trained model will be printed on the terminal. The output displays how the model performs during different epochs. It wll also produce first graph- Accuracy vs Epoches- which would show how the accuracy was improving during training. This graph will also be saved inside the src file, where the commands are being run. In the report, graph can be seen in Figure 10. Once you X out the graph, command will output accuracy, Precision and Recall scores. These values will replicate the data in Table 2 in report. Once all the output is printed, scroll up to find the total time taken to train the model and to predict the output for unseen code. The time taken is shown in report in Table 3. Print outs would also provide the total number of examples those were run the model on. Statistics of total number of file will repicate the information provided in Table 1. 

For Empty BlockStatement, change keyword IncorrectReturnStatment to IncorrectIfStatement in command. You should also update the last two arguments that provides the names of files those were saved from first step

Instructions to run on big corpus:
The only differece between small corpus and big corpus is that the big corpus has intense amount of data. We would run the same two commands described above, but will replace the programs_50 to programs_all. Expect the first command to produce more than one parsed file as a result. To organize the out put files, you can place them in a folder. To run the second command, the .json files would be changes to folder_name/* to point it to the newly saved folder. 


