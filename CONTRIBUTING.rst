Get in Touch!
===============

You have found a bug in pandapower or have a suggestion for a new functionality? Then get in touch with us by opening up an issue on the pandapower issue board to discuss possible new developments with the community and the maintainers.


Setup your git repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Note**: *The following setup is just a suggestion of how to setup your repository and is supposed to make contributing easier, especially for newcomers. If you have a different setup that you are more comfortable with, you do not have to adopt this setup.*

If you want to contribute for the first time, you can set up your environment like this:

#. If you have not done it yet: install git and create a github account
#. Create a fork of the official pandapower repository by clicking on "Fork" in the official pandapower repository (see https://help.github.com/articles/fork-a-repo/)
#. Clone the forked repository to your local machine: ::

    git clone https://github.com/YOUR-USERNAME/pandapower.git

#. Copy the following configuration at the bottom of to the pandapower/.git/config file (the .git folder is hidden, so you might have to enable showing hidden folders) and insert your github username: ::

    [remote "origin"]
        url = https://github.com/e2nIEE/pandapower.git
        fetch = +refs/heads/*:refs/remotes/pp/*
        pushurl = https://github.com/YOUR-USERNAME/pandapower.git
    [remote "pp"]
        url = https://github.com/e2nIEE/pandapower.git
        fetch = +refs/heads/*:refs/remotes/pp/*
    [remote "pp_fork"]
        url = https://github.com/YOUR-USERNAME/pandapower.git
        fetch = +refs/heads/*:refs/remotes/pp_fork/*
    [branch "develop"]
        remote = origin
        merge = refs/heads/develop

The develop branch is now configured to automatically track the official pandapower develop branch. So if you are on the develop branch and use: ::

    git pull

your local repository will be updated with the newest changes in the official pandapower repository.

Since you cannot push directly to the official pandapower repository, if you are on develop and do: ::

    git push

your push is by default routed to your own fork instead of the official pandapower repository with the setting as defined above.

If this is to implicit for you, you can always explicitely use the remotes "pp" and "pp_fork" to push and pull from the different repositories: ::

    git pull pp develop
    git push pp_fork develop

Contribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All contributions to the pandapower repository are made through pull requests to the develop branch. You can either submit a pull request from the develop branch of your fork or create a special feature branch that you keep the changes on. A feature branch is the way to go if you have multiple issues that you are working on in parallel and want to submit with seperate pull requests. If you only have small, one-time changes to submit, you can also use the develop branch to submit your pull request.

**Note**: *The following guide assumes the remotes are set up as described above. If you have a different setup, you will have to adapt the commands accordingly.*

Contribute from your develop branch
------------------------------------

#. Check out the develop branch on your local machine: ::

    git checkout develop

#. Update your local copy to the most recent version of the pandpower develop branch: ::

    git pull

#. Make changes in the code

#. Add and commit your changes: ::

    git add --all
    git commit -m "commit message"

   If there is an open issue that the commit belongs to, reference the issue in the commit message, for example for issue 3: ::

    git commit -m "commit message #3"

#. Push your changes to your fork: ::

    git push

#. Put in a Pull request to the main repository: https://help.github.com/articles/creating-a-pull-request-from-a-fork/

#. If you want to amend the pull request (for example because tests are failing in github actions, or because the community/maintainers have asked for modifications), simply push more commits to the branch: ::

    git add --all
    git commit -m "I have updated the pull request after discussions #3"
    git push

   The pull request will be automatically updated.

Contribute from a feature branch
------------------------------------

#. Check out the develop branch on your local machine: ::

    git checkout develop

#. Update your local copy to the most recent version of the pandpower develop branch: ::

    git pull

#. Create a new feature branch: ::

    git checkout -b my_branch

#. Make changes in the code

#. Add and commit your change: ::

    git add --all
    git commit -m "commit message"

   If there is an open issue that the commit belongs to, reference the issue in the commit message, for example for issue 3: ::

    git commit -m "commit message #3"

#. Push your changes to your fork: ::

    git push -u pp_fork my_branch

   this pushes the new branch to your fork and also sets up the remote tracking.

#. Put in a Pull request to the official repository (see https://help.github.com/articles/creating-a-pull-request-from-a-fork/)

#. If you want to amend the pull request (for example because tests are failing in github actions, or because the community/maintainers have asked for modifications), simply push more commits to the branch. Since the remote tracking branch has been set up, this is as easy as: ::

    git add --all
    git commit -m "I have updated the pull request after discussions #3"
    git push

#. If the pull request was merged and you don't expect further development on this feature, you can delete the feature branch to keep your repository clean.

Test Suite
~~~~~~~~~~~~~~~~

pandapower uses pytest for automatic software testing.

Making sure you don't break anything
---------------------------------------

If you make changes to pandapower that you plan to submit, first make sure that all tests are still passing. You can do this locally with: ::

    import pandapower.test
    pandapower.test.run_all_tests()

When you submit a pull request, github actions will run the same tests with actual, by pandapower supported Python versions. In most cases, if tests pass for you locally, they will also pass on github actions. But it can also happen that the tests pass for you locally, but still fail on github actions, because the new code is not compatible with all Python versions.
In this case you will have to update your pull request until the tests pass in all Python versions. Pull requests that lead to failing tests will not be accepted.


Adding Tests for new functionality
-----------------------------------

If you have added new functionality, you should also add a new function that tests this functionality. pytest automatically detects all functions in the pandapower/test folder that start with 'test' and are located in a file that also starts with 'test' as relevant test cases.


Say you have added a new function that for some doubles the rated power of a grid and added it to pandapower/toolbox.py: ::

    def double_rated_power(net):
       net.sn_kva *= 2

You would then add a new test to the test suite. In this case test/api/test_toolbox.py is the file that contains the tests for the toolbox functions. You then add a new test function to this file: ::

    def test_double_rated_power():
       init_sn_kva = 100
       net = pp.create_empty_network(sn_kva=init_sn_kva)
       assert net.sn_kva == init_sn_kva
       pp.double_rated_power(net)
       assert net.sn_kva == init_sn_kva*2

This function is now automatically detected by pytest as part of the test suite and will be tested by Travis CI in all Python versions. If someone later changes your new toolbox function and introduces a bug like this: ::

    def double_rated_power(net):
       net.sn_kva *= 3

The introduced test will fail. If this change would be submitted, the pull request would not be accepted, as it leads to failing tests in Travis CI. In that way, tests ensure the continuing integrity of the development and ensure that no functionality is inadvertently broken.

Tests with pytest can be more complex than the simple example above. For how to handle e.g. pytest fixtures, xfailing tests etc. refer to the documentation of pytest.
