import numpy as np
import random

"""
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
"""



def flatten(llist):
    return [item for sublist in llist for item in sublist]



def get_minibatches(dataset_address, minibatch_size = 100):
    data_size = sum(1 for line in open(dataset_address + ".span"))
    question_id_file = open(dataset_address + ".ids.question", 'r')
    context_id_file = open(dataset_address + ".ids.context", 'r')
    answer_file = open(dataset_address + ".span", 'r')
    file_eof = False
    
    while not file_eof:
        
        questions = []
        contexts = []
        start_answers = []
        end_answers = []

        minibatch_counter = 0
        while minibatch_counter < minibatch_size:
            question_line = question_id_file.readline()
            context_line = context_id_file.readline()
            answer_line = answer_file.readline()

            if (question_line == ""): # EOF
                file_eof = True
                break

            context = map(int, context_line.split(" "))
            question = map(int, question_line.split(" "))
            answer = map(int, answer_line.split(" "))
            
            if answer[0] >= max_context_length or answer[1] >= max_context_length:
                continue;
            
            contexts.append(context)            
            questions.append(question)            
            start_answers.append([answer[0]])
            end_answers.append([answer[1]])
            minibatch_counter += 1

        minibatch = [questions, contexts, start_answers, end_answers]
        yield minibatch # yield, return as a generator
    
    question_id_file.close()
    context_id_file.close()
    answer_file.close()



def get_sample(dataset_address, sample_size=100, max_context_length): 
    
    data_size = sum(1 for line in open(dataset_address + ".span"))

    #question_id_file = open(dataset_address + ".ids.question", 'r')
    context_id_file = open(dataset_address + ".ids.context", 'r')
    answer_file = open(dataset_address + ".span", 'r')    

    indices = set(random.sample(range(0, data_size), sample_size))
    max_index = max(indices)
    
    contexts = []
    questions = []
    start_answers = []
    end_answers = []
    num_samples = sample_size
    line_number = 0
    for question_line in open(dataset_address + ".ids.question", 'r'):
        if line_number > max_index:
            break
        context_line = context_id_file.readline()
        answer_line = answer_file.readline()
        if line_number in indices:
            # This is one of the selected rows, add it to the minibatch
            question = map(int, question_line.split(" "))
            context = map(int, context_line.split(" "))
            answer = map(int, answer_line.split(" "))
            if answer[0] >= max_context_length or answer[1] >= max_context_length:
                num_samples -= 1
                line_number += 1
                continue
            questions.append(question)
            contexts.append(context)
            start_answers.append([answer[0]])
            end_answers.append([answer[1]])
        line_number += 1

    context_id_file.close()
    answer_file.close()
    return [[questions, contexts, start_answers, end_answers], num_samples]
