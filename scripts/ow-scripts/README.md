# OpenWhisk Scripts

These scripts build and deploy OpenWhisk in multiple Docker containers on a multi-node setup. We successfully ran these scripts on bare metal nodes in the Chameleon cluster on TACC running an Ubuntu 18.04 Bionic Image. 

## Setting Up OpenWhisk for the First Time
1. First, set up docker on all machines. Use `install-docker.sh` to do so. I suggest you run each command in the script manually on each node, as you will need to logout/login (line 39 in the script) to ensure that your user is added to the docker group.
2. Next, install packages that are necessary using `wsk-setup.sh`. This will probably need to be done manually by copying the lines `4-15`. 
3. Clone OpenWhisk manually. Change some of the ansible scripts to set up the distributed OpenWhisk system:
    - Open `openwshisk/ansible/environments/local/hosts.j2.ini` and change everything except the invoker IP to the current machine you're on. This will be your controller.
    - Then change the invoker ip to the second node's IP address. Add any more invoker machines that you might have.
    - Finally, make sure that all your machines have SSH access without the need for a password. Create public keys for every machine and add them to `~/.ssh/authorized_hosts` in every other machine. This can be done using `ssh-keygen`.
4. Go through the rest of the steps in `wsk-setup.sh`.
5. Next you will need top change the `db_host` variable in `openwhisk/ansible/db_local.ini` to the current machine you're on.  
6. Now go through `wsk-deploy.sh`. Everything should work out the gate.
7. Finally, go through `wsk-cli-setup.sh`. Once again, everything should work out the gate. 

## Testing if Set Up was Successful
1. Outside of the `openwhisk` directory, check if the `wsk` cli is working (`wsk -help` should work).
2. Create a hello.js file and add the following contents (again, outside `openwhisk`).
```
/**
 * Hello world as an OpenWhisk action.
 */
function main(params) {
    var name = params.name || 'World';
    return {payload:  'Hello, ' + name + '!'};
}
```
3. Create an action named `hello` using `wsk`
```
wsk action create hello hello.js
```
4. Invoke the action. Your output should be as shown below.
```
wsk action invoke hello --result
```
```
{
    "payload": "Hello, World!"
}
```

## Clean OpenWhisk Containers
To clean OpenWhisk containers that were created while setting up, run `wsk-clean.sh`. The following docker containers will not be seen anymore upon `docker ps`:
    - nginx
    - invoker
    - controller
    - kafka
    - zookeeper

You will see the following containers still up:
    - apigateway
    - redis
    - couchdb

## Wipe OpenWhisk Completely
To completely wipe OpenWhisk, APIGateway, and the CouchDB database, run `wsk-wipe.sh`. 
