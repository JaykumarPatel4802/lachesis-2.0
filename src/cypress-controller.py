import grpc
import queue
import threading
import time
import math
from concurrent import futures

from generated import cypress_pb2_grpc, cypress_pb2

invocation_queue = queue.Queue()

class Cypress(cypress_pb2_grpc.CypressServicer):
    def __init__(self):
        self.function_queues = {}

    def Register(self, request, context):
        # Setup function metadata part of registration command
        function_metadata_string = ''
        for metadata in request.function_metadata:
            split_metadata = metadata.split(':')
            key = split_metadata[0]
            value = ''
            for i in range(1, len(split_metadata)):
                value += split_metadata[i] + ":"
            value = value[0:-1]
            function_metadata_string += '--{} {} '.format(key, value)

        # Setup function parameters of registration command
        parameter_string = ''
        for parameter in request.parameters:
            split_parameter = parameter.split(':')
            key = split_parameter[0]
            value = ''
            for i in range(1, len(split_parameter)):
                value += split_parameter[i] + ":"
            value = value[0:-1]
            parameter_string += '--param {} {} '.format(key, value)
        
        # Create final registration command, one per cpu core
        for cpu in range(1, MAX_CPU_ALLOWED + 1):
            for mem_class in range(1, MAX_MEM_CLASSES + 1):
                memory = mem_class * 128
                fxn_registration_command = 'cd {}; wsk -i action update {}_{}_{} {}.py --cpu {} --memory {} {} {}\n'.format(request.function_path, request.function, cpu, memory, request.function, cpu, memory, function_metadata_string, parameter_string)
                tmp = subprocess.Popen(fxn_registration_command, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                fxn_reg_out, fxn_reg_err = tmp.communicate()
                fxn_reg_out = fxn_reg_out.decode()
                fxn_reg_err = fxn_reg_err.decode()
                if 'ok: updated action' not in fxn_reg_out:
                    return cypress_pb2.Reply(status='FAILURE', message='failed to register function {} with cpu {} and memory {}. Output was: {}. Eror was: {}.'.format(request.function, cpu, memory, fxn_reg_out, fxn_reg_err))
            
        return cypress_pb2.Reply(status='SUCCESS', message='successfully registered function {} with all {} cpu levels and {} memory levels'.format(request.function, MAX_CPU_ALLOWED, MAX_MEM_CLASSES))

    def Invoke(self, request, context):
        function = request.function

        # Check if the queue exists for the function, create one if not
        if function not in self.function_queues:
            self.function_queues[function] = queue.Queue()
            # Create a thread to monitor the new queue
            thread = threading.Thread(target=self.monitor_queue, args=(function,))
            thread.daemon = True
            thread.start()
        
        current_time = time.time()
        item = (current_time, request.parameters, request.slo, request.batch_size)
        self.function_queues[function].put(item)

        return cypress_pb2.Reply(status='SUCCESS', message=f'Queued {request.function} with {request.slo} SLO and {request.parameters}')

    def monitor_queue(self, function):
        while True:
            func_queue = self.function_queues[function]
            try:
                item = func_queue.get(block=True)
                time, parameters, slo, batch_size = item
                batch = []
                batch.append((function, parameters, slo, batch_size))
                while True:
                    updated_queue = self.function_queues[function]
                    next_item = updated_queue.get(block=True)
                    next_time, next_parameters, next_slo, next_batch_size = next_item
                    if next_time > time + 2:
                        invocation_queue.put(batch)
                        time, parameters, slo, batch_size = next_time, next_parameters, next_slo, next_batch_size
                        batch = []
                        batch.append((function, parameters, slo, batch_size))
                    else:
                       batch.append((function, next_parameters, next_slo, next_batch_size)) 
            except queue.Empty:
                pass

def process_invocation_queue():
    while True:
        batch = invocation_queue.get()
        invocations = {}
        for item in batch:
            function = item[0]
            key = math.floor(item[3])
            if key in invocations:
                invocations[key].append((item[0], item[1]))
            else:
                invocations[key] = [(item[0], item[1])]
        print(invocations)
        print('------------------------')

def run_server():
    # Start the shared queue processing thread
    invocation_queue_thread = threading.Thread(target=process_invocation_queue)
    invocation_queue_thread.daemon = True
    invocation_queue_thread.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cypress_pb2_grpc.add_CypressServicer_to_server(Cypress(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Cypress server is up and running, ready for your request!')
    server.wait_for_termination()

if __name__=='__main__':
    run_server()