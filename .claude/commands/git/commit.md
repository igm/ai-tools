I want you to commit the changes to the current branch. If there are staged changes commit them. If there are no staged changes commit all changes.

Important: Before committing ensure all the tests pass.  If tests do not pass make all necessary changes to make them pass.  

If needed also group changes in multiple commits in case the changes are not related.


## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Code style changes (formatting, semicolons, etc)
- `refactor` - Code refactoring (neither feature nor fix)
- `perf` - Performance improvements
- `test` - Adding or updating tests
- `build` - Build system or dependencies
- `ci` - CI/CD changes
- `chore` - Other changes

**Examples:**
- `feat: add NVDA trend analysis endpoint`
- `fix(auth): correct JWT token validation`
- `docs: update deployment guide with ECR steps`

**Guidelines:**
- Use imperative mood ("add" not "added")
- Limit first line to 72 chars
- Reference issues in footer: `Closes #123`
- Include plan details in body if applicable
- Explain WHY changes were made, not just WHAT 
