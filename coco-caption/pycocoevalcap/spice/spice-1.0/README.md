Semantic Propositional Image Caption Evaluation (SPICE)
===================

Evaluation code for machine-generated image captions.

## Requirements ##
- java 1.8.0+

## Dependencies ##
- Stanford [CoreNLP](http://stanfordnlp.github.io/CoreNLP/) 3.6.0
- Stanford [Scene Graph Parser](http://nlp.stanford.edu/software/scenegraph-parser.shtml)
- [Meteor](http://www.cs.cmu.edu/~alavie/METEOR/) 1.5 (for synset matching)

## Usage ##

To run SPICE, call the following:

    java -Xmx8G -jar spice-*.jar

Running SPICE with no arguments prints the following help message:

    SPICE version 1
    
    Usage: java -Xmx8G -jar spice-*.jar <input.json> [options]
    
    Options:
    -out <outfile>                   Output json scores and tuples data to <outfile>
    -cache <dir>                     Set directory for caching reference caption parses
    -threads <num>                   Defaults to the number of processors
    -detailed                        Include propositions for each caption in json output.
    -noSynsets                       Disable METEOR-based synonym matching
    -subset                          Report results in <outfile> for various semantic tuple subsets
    -silent                          Disable stdout results
    
    See README file for additional information and input format details

The input.json file should contain of an array of json objects, each representing a single caption and containing `image_id`, `test` and `refs` fields. See `example_input.json`

It is recommended to provide a path to an empty directory in the `-cache` argument as it makes repeated evaluations much faster.

## Build ##
To build SPICE and its dependencies from source, and run tests, use Maven with the following command: `mvn clean verify`. The jar file spice-*.jar will be created in the target directory, with required dependencies in target/src.

Building SPICE from source is NOT required as precompiled jar files are available on the [project page](http://panderson.me/spice).

## References ##
If you report SPICE scores, please cite the SPICE paper:
- [Semantic Propositional Image Caption Evaluation (SPICE)](http://panderson.me/images/SPICE.pdf) 
- [bibtex](http://panderson.me/images/SPICE.bib)

## Developers ##
- [Peter Anderson](http://panderson.me) (Australian National University) (peter.anderson@anu.edu.au)

## Acknowledgements ##
- This work is based on the [SceneGraphParser](http://nlp.stanford.edu/software/scenegraph-parser.shtml) developed by [Sebastian Schuster](http://sebschu.com/) (Stanford).
- We re-use the Wordnet synset matching code from [Meteor 1.5](http://www.cs.cmu.edu/~alavie/METEOR/) to identify synonyms.

## License ##
- GNU AGPL v3
