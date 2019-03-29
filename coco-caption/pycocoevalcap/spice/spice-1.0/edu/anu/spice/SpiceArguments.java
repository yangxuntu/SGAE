/*
 * Copyright (c) 2016, Peter Anderson <peter.anderson@anu.edu.au>
 *
 * This file is part of Semantic Propositional Image Caption Evaluation
 * (SPICE).
 * 
 * SPICE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * SPICE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.

 * You should have received a copy of the GNU Affero General Public
 * License along with SPICE.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

package edu.anu.spice;

public class SpiceArguments {

	public String inputPath;
	public String outputPath;
	public String cache;
	public int numThreads;
	public Boolean detailed;
	public Boolean synsets;
	public Boolean tupleSubsets;
	public Boolean silent;

	SpiceArguments() {
		inputPath = null;
		outputPath = null;
		cache = null;
		numThreads = Runtime.getRuntime().availableProcessors();
		detailed = false;
		synsets = true;
		tupleSubsets = false;
		silent = false;
	}

	static void printUsage() {
		System.err.println("SPICE version 1");
		System.err.println();
		System.err.println("Usage: java -Xmx8G -jar spice-*.jar <input.json> [options]");
		System.err.println();
		System.err.println("Options:");
		System.err.println("-out <outfile>                   Output json scores and tuples data to <outfile>");
		System.err.println("-cache <dir>                     Set directory for caching reference caption parses");
		System.err.println("-threads <num>                   Defaults to the number of processors");
		System.err.println("-detailed                        Include propositions for each caption in json output.");
		System.err.println("-noSynsets                       Disable METEOR-based synonym matching");
		System.err.println("-subset                       	 Report results in <outfile> for various semantic tuple subsets");
		System.err.println("-silent                       	 Disable stdout results");
		System.err.println();
		System.err.println("See README file for additional information and input format details");
	}

	SpiceArguments(String[] args) {
		this();
		this.inputPath = args[0];
		int curArg = 1;
		while (curArg < args.length) {
			if (args[curArg].equals("-out")) {
				this.outputPath = args[curArg + 1];
				curArg += 2;
			} else if (args[curArg].equals("-cache")) {
				this.cache = args[curArg + 1];
				curArg += 2;				
			} else if (args[curArg].equals("-threads")) {
				this.numThreads = Integer.parseInt(args[curArg + 1]);
				curArg += 2;
			} else if (args[curArg].equals("-detailed")) {
				this.detailed = true;
				curArg += 1;
			} else if (args[curArg].equals("-noSynsets")) {
				this.synsets = false;
				curArg += 1;
			} else if (args[curArg].equals("-subset")) {
				this.tupleSubsets = true;
				curArg += 1;
			} else if (args[curArg].equals("-silent")) {
				this.silent = true;
				curArg += 1;
			} else {
				System.err.println("Unknown option \"" + args[curArg] + "\"");
				System.exit(1);
			}
		}
	}

}
