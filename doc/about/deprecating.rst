Best practise process of deprecating code
------------------------------------------

There are many ways to handle deprecated code, the following describes how we would like to implement this in pandapower.

If existing pandapower functionality should be improved which may damage applications of users, deprecation warning should be implemented to notify the user to update the usage of pandapower.
**Deprecation warnings** should exist from one minor version release (at least) until the next major or minor version release.
When announcing pandapower functionality changes, always provide the new desired functionality instantly.

This figure illustrates a time sequence of a process with deprecation warnings, assuming that the need for revision has been determined and implemented at pandapower version `v2.10.0`.

.. image:: /pics/deprecating/deprecating.svg
		:width: 500em
		:align: center

Proposed steps for contributers and reviewers:

#. Provide two pull requests (PR1 and PR2) including
    #. final code that should remain in the long-term
        * if input parameter names have been changed, these can be catched and notified using :code:`raise DeprecationWarning("Notification message")`
    #. interim code to be contained in the master branch from the next to the next but one release version
        * to implement deprecation warnings type :code:`warnings.warn("Notification message", category=DeprecationWarning)` (:code:`category=FutureWarning` is also possible; don't forget :code:`import warnings`)
        * the new functionality should be available but if the user does not pass arguments that explicitely call the new functionality, the old functionality should be evaluated plus sending DeprecationWarnings
        * make sure to add new tests for the new functionality
        * please add :code:`assert Version(pp.__version__) < Version('2.12')` with :code:`from packaging.version import Version` to the tests of the deprecated functionality (assuming that the latest pandapower version is `v2.10.x` ; tests of the deprecated code should be removed with PR1)
#. Reviewer: don't approve the PRs as long as both PRs are correct and complete
#. The person merging PR2 is responsible to merge PR1 after the next release

As an example, please have a look at the revision of the function `pandapower.toolbox.merge_nets()`.
It has been revised by `PR2 <https://github.com/e2nIEE/pandapower/pull/1764>`_ temporarly. This state is available from pandapower version `v2.10.0` to `v2.11.1`.
With `v2.12.0` the final code, merged via `PR1 <https://github.com/e2nIEE/pandapower/pull/1765>`_, is available.

For suggestions on revising this process, `Issue 1760 <https://github.com/e2nIEE/pandapower/issues/1760>`_ can be reopened.
