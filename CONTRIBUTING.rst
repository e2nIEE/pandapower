
Setup
----------

#. If you have not done it yet: create a github account
#. Create a fork of the official pandapower repository (see https://help.github.com/articles/fork-a-repo/)  
#. Clone the forked repository to your local machine
#. Copy the following configuration at the bottom of to the pandapower/.git/gitconfig file (the .git folder is hidden, so you might have to enable showing hidden folders) and insert your github username:

.. code-block:: 

    [remote "origin"]
        url = https://github.com/e2nIEE/pandapower.git
        fetch = +refs/heads/*:refs/remotes/pp/*
        pushurl = https://github.com/YOURGITHUBUSERNAME/pandapower.git
    [remote "pp"]
        url = https://github.com/e2nIEE/pandapower.git
        fetch = +refs/heads/*:refs/remotes/pp/*
    [remote "pp_fork"]
        url = https://github.com/YOURGITHUBUSERNAME/pandapower.git
        fetch = +refs/heads/*:refs/remotes/pp_fork/*
    [branch "develop"]
        remote = origin
        merge = refs/heads/develop
        
The develop branch is now configured to automatically track the official pandapower develop branch. So if you pull the develop branch, it will by default pull from the official repository.
Since you cannot push directly to the official pandapower repository, pushes are by default routed to your own fork instead of the official pandapower repository.

Contribute from your develop branch
------------------------------------

#. Check out the develop branch on your local machine:

    git checkout develop

#. Update your local copy to the most recent version of the pandpower develop branch

    git pull

#. Make changes in the code

#. Add and commit your changes

    git add --all

    git commit -m"commit message"

#. Push your changes to your fork:

    git push

#. Put in a Pull request to the main repository: https://help.github.com/articles/creating-a-pull-request-from-a-fork/

Contribute from your develop branch
------------------------------------

#. Check out the develop branch on your local machine:

    git checkout develop

#. Update your local copy to the most recent version of the pandpower develop branch

    git pull

#. Create a new branch:

    git checkout -b my_branch
    
#. Make changes in the code

#. Add and commit your changes

    git add --all

    git commit -m"commit message"

#. Push your changes to your fork:

    git push -u pp_fork my_branch
    
   this pushes the new branch to your fork and also sets up the remote tracking. So if you make more commits and push them, you only need to do:

    git push
   

#. Put in a Pull request to the main repository: https://help.github.com/articles/creating-a-pull-request-from-a-fork/


