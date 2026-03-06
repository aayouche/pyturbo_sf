Contributing to PyTurbo_SF
=========================

We welcome contributions to PyTurbo_SF! This guide will help you get started with contributing to the project.

Getting Started
---------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork and Clone the Repository**

   .. code-block:: bash

      # Fork the repository on GitHub, then clone your fork
      git clone https://github.com/yourusername/pyturbo_sf.git
      cd pyturbo_sf

2. **Create a Development Environment**

   .. code-block:: bash

      # Using conda (recommended)
      conda create -n pyturbo_dev python=3.12
      conda activate pyturbo_dev
      
      # Or using venv
      python -m venv pyturbo_dev
      source pyturbo_dev/bin/activate  # Linux/Mac
      # pyturbo_dev\Scripts\activate  # Windows

3. **Install in Development Mode with Dev Dependencies**

   .. code-block:: bash

      # Install package in editable mode with dev dependencies
      pip install -e ".[dev]"

4. **Verify Installation**

   .. code-block:: bash

      # Run tests to ensure everything works
      pytest tests/
      
      # Check code style
      flake8 pyturbo_sf/
      
      # Check type hints
      mypy pyturbo_sf/

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Create a Feature Branch**

   .. code-block:: bash

      git checkout -b feature/your-feature-name
      # or
      git checkout -b bugfix/issue-number

2. **Make Your Changes**

   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**

   .. code-block:: bash

      # Run the full test suite
      pytest tests/ -v
      
      # Run specific test categories
      pytest tests/test_1d.py -v
      pytest tests/test_2d.py -v
      pytest tests/test_3d.py -v
      
      # Check test coverage
      pytest --cov=pyturbo_sf tests/

4. **Commit and Push**

   .. code-block:: bash

      git add .
      git commit -m "Add feature: your descriptive commit message"
      git push origin feature/your-feature-name

5. **Create a Pull Request**

   - Go to GitHub and create a pull request
   - Describe your changes clearly
   - Reference any related issues

Types of Contributions
----------------------

Code Contributions
~~~~~~~~~~~~~~~~~~

**New Features:**
   - Additional structure function types
   - Performance optimizations
   - New analysis utilities
   - Enhanced data format support

**Bug Fixes:**
   - Fix calculation errors
   - Improve error handling
   - Platform-specific issues
   - Memory management improvements

**Performance Improvements:**
   - Algorithm optimizations
   - Parallel computing enhancements
   - Memory usage reductions
   - Caching mechanisms

Documentation Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Documentation Types:**
   - API documentation improvements
   - Tutorial notebooks
   - Example scripts
   - Performance guides
   - Troubleshooting guides

**Areas Needing Documentation:**
   - Real-world use cases
   - Advanced applications
   - Platform-specific instructions
   - Integration with other tools

Testing Contributions
~~~~~~~~~~~~~~~~~~~~~

**Test Categories:**
   - Unit tests for individual functions
   - Integration tests for workflows
   - Performance benchmarks
   - Cross-platform compatibility tests
   - Regression tests

**Test Data:**
   - Synthetic test datasets
   - Reference solutions
   - Edge case scenarios
   - Large dataset tests


Pull Request Guidelines
-----------------------

Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~~

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] No breaking changes (or clearly documented)

Pull Request Template
~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes and motivation.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Other (please describe)
   
   ## Testing
   - [ ] All existing tests pass
   - [ ] New tests added for new functionality
   - [ ] Manual testing performed
   
   ## Documentation
   - [ ] Docstrings updated
   - [ ] README updated (if needed)
   - [ ] Examples updated (if needed)
   
   ## Performance Impact
   - [ ] No performance impact
   - [ ] Performance improvement
   - [ ] Performance regression (justified)
   
   ## Additional Notes
   Any additional information or context.

Review Process
~~~~~~~~~~~~~~

1. **Automated Checks:** 
   - Code style (flake8)
   - Type checking (mypy)
   - Tests (pytest)
   - Coverage (codecov)

2. **Manual Review:**
   - Code quality and clarity
   - Algorithm correctness
   - Performance implications
   - Documentation quality

3. **Merge Requirements:**
   - All checks pass
   - At least one approving review
   - No merge conflicts
   - Up-to-date with main branch

Community Guidelines
--------------------

Communication
~~~~~~~~~~~~~

**Channels:**
   - GitHub Issues: Bug reports, feature requests
   - GitHub Discussions: General questions, ideas
   - Pull Requests: Code contributions

**Guidelines:**
   - Be respectful and inclusive
   - Provide clear, detailed descriptions
   - Search existing issues before creating new ones
   - Use appropriate labels and templates

Issue Reporting
~~~~~~~~~~~~~~~

**Bug Reports Should Include:**
   - Python version and platform
   - PyTurbo_SF version
   - Minimal reproducible example
   - Expected vs actual behavior
   - Full error traceback (if applicable)

**Feature Requests Should Include:**
   - Clear description of the feature
   - Use case and motivation
   - Proposed API (if applicable)
   - Implementation suggestions (optional)

Recognition
-----------

Contributors will be recognized in:

- AUTHORS file
- Release notes
- Documentation acknowledgments
- Annual contributor highlights

**Types of Recognition:**
   - Code contributors
   - Documentation contributors
   - Bug reporters
   - Community helpers
   - Reviewers

Development Roadmap
-------------------

Current Priorities
~~~~~~~~~~~~~~~~~

1. **Performance Optimization:**
   - GPU acceleration investigation
   - Memory usage improvements
   - Advanced parallel algorithms

2. **New Features:**
   - Additional structure function types
   - Enhanced error analysis
   - Better visualization tools

3. **Usability:**
   - Improved documentation
   - More examples and tutorials
   - Better error messages

4. **Testing:**
   - Expanded test coverage
   - Performance benchmarks
   - Cross-platform testing

How to Help
~~~~~~~~~~~

**Immediate Needs:**
   - Test on different platforms
   - Documentation improvements
   - Performance benchmarking
   - Bug reports and fixes

**Future Opportunities:**
   - GPU implementation
   - Advanced algorithms
   - Integration with other tools
   - Educational materials

Questions and Support
--------------------

If you have questions about contributing:

1. Check existing documentation and issues
2. Ask in GitHub Discussions
3. Contact maintainers directly (for sensitive issues)
