import numpy as np

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def get_minibatches(data, minibatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)

def flatten(llist):
    return [item for sublist in llist for item in sublist]


#remember to take a second look at evaluate_answer in qa_model. that function needs slightly
# different values from get_minibatches

def get_minibatches(dataset_address, minibatch_size = 100):
    question_id_file = open(dataset_address + ".ids.question", 'r')
    context_id_file = open(dataset_address + ".ids.context", 'r')
    answer_file = open(dataset_address + ".span", 'r')
    
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        
        questions = []
        contexts = []
        start_answers = []
        end_answers = []
        
        for _ in range(minibatch_size):
            question = map(int, question_id_file.readline().split(" "))
            questions.append(question)

            context = map(int, context_id_file.readline().split(" "))
            contexts.append(context)

            answer = map(int, answer_file.readline().split(" "))
            start_answers.append([answer[0]])
            end_answers.append([answer[1]])

        minibatch = [questions, contexts, start_answers, end_answers]
        yield minibatch # yield, return as a generator
    
    question_id_file.close()
    context_id_file.close()
    answer_file.close()

def get_sample(dataset_address, sample_size):
    
    data_size = sum(1 for line in open(dataset_address + ".span"))
    question_id_file = open(dataset_address + ".ids.question", 'r')
    context_id_file = open(dataset_address + ".ids.context", 'r')
    answer_file = open(dataset_address + ".span", 'r')    
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    indices = indices[:sample_size]
    indices_rev = {indices[i]:i for i in range(sample_size)}
    indices.sort()
    
    questions = [0] * sample_size
    start_answers = [0] * sample_size
    end_answers = [0] * sample_size
    indices_counter = 0
    line_counter = 0
    
    while indices_counter < sample_size:
        question_line = question_id_file.readline()
        context_line = context_id_file.readline()
        answer_line = answer_file.readline()
        if indices[indices_counter] == line_counter:
            question = map(int, question_line.split(" "))
            context = map(int, context_line.split(" "))
            answer = map(int, answer_line.split(" "))
            questions[indices_rev[indices[indices_counter]]] = question
            contexts[indices_rev[indices[indices_counter]]] = context
            start_answers[indices_rev[indices[indices_counter]]] = [answer[0]]
            end_answers[indices_rev[indices[indices_counter]]] = [answer[1]]
            indices_counter += 1
        line_counter += 1
    
    final_sample = [questions, contexts, start_answers, end_answers]
    question_id_file.close()
    context_id_file.close()
    answer_file.close()
    return final_sample 
