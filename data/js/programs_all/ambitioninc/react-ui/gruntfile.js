'use strict';

module.exports = function(grunt) {

    /**
     * @method registerTasks
     * Registers grunt tasks for command line usage.
     */
    function registerTasks() {
        grunt.registerTask('buildSrc', ['clean', 'react']);
        grunt.registerTask('build', ['buildSrc', 'browserify', 'uglify']);
        grunt.registerTask('test', ['clean', 'react', 'jshint', 'mochaTest', 'reportCoverage']);
        grunt.registerTask('watch', ['watch']);
    }

    /**
     * @method registerReportCoverage
     * Registers a task that can read the blanket coverage report.
     */
    function registerReportCoverage() {
        grunt.registerMultiTask('reportCoverage', 'Reads and reports a coverage json file', function() {
            var result = grunt.file.readJSON('.coverage/coverage.json');

            if (result.coverage < 100) {
                grunt.fail.warn(
                    'Expected coverage to be 100%, but was ' + result.coverage.toFixed(2) + '%.\n' +
                    'See ".coverage/coverage.html" for details.'
                );
            } else {
                grunt.log.ok('Code coverage is 100%. Reward yourself with a burrito.');
            }
        });
    }

    /**
     * @method loadNpmTasks
     * Loads tasks from installed grunt plugins.
     */
    function loadNpmTasks() {
        var npmTasks = [
            'grunt-browserify',
            'grunt-contrib-clean',
            'grunt-contrib-jshint',
            'grunt-contrib-uglify',
            'grunt-contrib-watch',
            'grunt-mocha-test',
            'grunt-react'
        ];

        npmTasks.forEach(function(task) {
            grunt.loadNpmTasks(task);
        });
    }

    /**
     * @method mochaHarness
     * Sets up the environment for running Mocha tests.
     */
    function mochaHarness() {
        var jsdom = require('jsdom');

        //build the dom for tests
        global.window = jsdom.jsdom(
            '<!DOCTYPE html>' +
            '<html>' +
                '<head>' +
                    '<meta charset="utf-8">' +
                    '<title>Mocha Test</title>' +
                '</head>' +
                '<body>' +
                '</body>' +
            '</html>'
        ).parentWindow;
        global.document = global.window.document;
        global.navigator = global.window.navigator;
        global.React = require('react/addons');

        //cover all the compiled js files
        require('blanket')({pattern: 'transformed'});
    }

    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),

        /**
         * Builds the project for the browser.
         */
        browserify: {
            build: {
                files: {'build/react-ui.js': 'browserify.js'}
            },
        },

        /**
         * Removes temporary files generated by building and testing.
         * Should be called before any builds or tests.
         */
        clean: {
            coverage: ['.coverage'],
            grunt: ['.grunt'],
            transformed: ['transformed']
        },

        /**
         * Runs static analysis on the source code.
         * Must be preceeded by the {@link react} task.
         */
        jshint: {
            lint: {
                src: [
                    'gruntfile.js',
                    'transformed/*.js',
                    'transformed/**/*.js'
                ],
                options: {jshintrc: '.jshintrc'}
            }
        },

        /**
         * Runs tests and creates coverage files.
         * Creates both json and html coverage files.
         * Should be followed by the {@link readCoverage} task.
         */
        mochaTest: {
            test: {
                options: {
                    reporter: 'spec',
                    require: [mochaHarness]
                },
                src: [
                    'transformed/*.js',
                    'transformed/**/*.js'
                ]
            },
            jsonCoverage: {
                options: {
                    reporter: 'json-cov',
                    quiet: true,
                    captureFile: '.coverage/coverage.json'
                },
                src: [
                    'transformed/*.js',
                    'transformed/**/*.js'
                ]
            },
            htmlCoverage: {
                options: {
                    reporter: 'html-cov',
                    quiet: true,
                    captureFile: '.coverage/coverage.html'
                },
                src: [
                    'transformed/*.js',
                    'transformed/**/*.js'
                ]
            }
        },

        /**
         * Reads coverage and fails the task if not covered 100%.
         */
        reportCoverage: {files: ''},

        /**
         * Transforms React source to javascript.
         */
        react: {
            all: {
               files: [{
                    expand: true,
                    cwd: 'src',
                    src: ['*.js', '**/*.js'],
                    dest: 'transformed',
                    ext: '.js'
               }]
            }
        },

        /**
         * Minifies the browserify build.
         */
        uglify: {
            build: {
                files: {'build/react-ui.min.js': 'build/react-ui.js'}
            }
        },

        /**
         * Runs source and/or style builds when files are edited.
         */
        watch: {
            src: {
                files: ['src/*.js', 'src/**/*.js'],
                tasks: ['buildSrc']
            }
        }
    });

    registerTasks();
    registerReportCoverage();
    loadNpmTasks();
};