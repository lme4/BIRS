## Git for modifications  -- Doug Bates

1. Create a branch on github (main repo page; "top bar"; item "<n> branches")
   - using initials in the branch name is a good idea, say `mm-foo-bar`
2. Pull to your local repository: `git pull`
3. Check out your branch locally:
   - `git branch --track  mm-foo-bar`
   - `git checkout        mm-foo-bar`

4. Edit your branch;  stage, commit ..

5. Either
   - push your changes, *and* on _github_, create a pull request
	 (which will send e-mail to collaborators and create a nice discussion
	 page),

	 or

   - _rebase_ your branch to the updated master

