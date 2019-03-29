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

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import com.google.common.base.Stopwatch;

public class SpiceScorer {
	
	public SpiceStats stats;
	
	SpiceScorer(){
		stats = null;
	}

	public static void main(String[] args) {
		if (args.length < 1) {
			SpiceArguments.printUsage();
			System.exit(2);
		}
		SpiceArguments spiceArgs = new SpiceArguments(args);
		try {
			SpiceScorer scorer = new SpiceScorer();
			scorer.scoreBatch(spiceArgs);
		} catch (Exception ex) {
			System.err.println("Error: Could not score batched file input:");
			ex.printStackTrace();
			System.exit(1);
		}
	}

	public void scoreBatch(SpiceArguments args) throws IOException, ScriptException {
		Stopwatch timer = Stopwatch.createStarted();
		SpiceParser parser = new SpiceParser(args.cache, args.numThreads, args.synsets);
		
		// Build filters for tuple categories
		Map<String, TupleFilter> filters = new HashMap<String, TupleFilter>();
		if (args.tupleSubsets) {
			filters.put("Object", TupleFilter.objectFilter);
			filters.put("Attribute", TupleFilter.attributeFilter);
			filters.put("Relation", TupleFilter.relationFilter);
			filters.put("Cardinality", TupleFilter.cardinalityFilter);
			filters.put("Color", TupleFilter.colorFilter);
			filters.put("Size", TupleFilter.sizeFilter);
		}
		
		// Parse test and refs from input file
		ArrayList<Object> image_ids = new ArrayList<Object>();
		ArrayList<String> testCaptions = new ArrayList<String>();
		ArrayList<String> refCaptions = new ArrayList<String>();
		ArrayList<Integer> refChunks = new ArrayList<Integer>();
		JSONParser json = new JSONParser();
		JSONArray input;
		try {
			input = (JSONArray) json.parse(new FileReader(args.inputPath));
			for (Object o : input) {
			    JSONObject item = (JSONObject) o;
			    image_ids.add(item.get("image_id"));
			    testCaptions.add((String) item.get("test"));
			    JSONArray refs = (JSONArray) item.get("refs");
			    refChunks.add(refs.size());
			    for (Object ref : refs){
			    	refCaptions.add((String) ref);
			    }
			}
		} catch (ParseException e) {
			System.err.println("Could not read input: " + args.inputPath);
			System.err.println(e.toString());
			e.printStackTrace();
		}
		
		System.err.println("Parsing reference captions");
		List<SceneGraph> refSgs = parser.parseCaptions(refCaptions, refChunks);
		System.err.println("Parsing test captions");		
		List<SceneGraph> testSgs = parser.parseCaptions(testCaptions);
		
		this.stats = new SpiceStats(filters, args.detailed);
		for (int i=0; i<testSgs.size(); ++i) {
			this.stats.score(image_ids.get(i), testSgs.get(i), refSgs.get(i), args.synsets);
		}
		if (!args.silent){
			System.out.println(this.stats.toString());
		}
		
		if (args.outputPath != null) {
			BufferedWriter outputWriter = new BufferedWriter(new FileWriter(args.outputPath));
			
			// Pretty print output using javascript
			String jsonStringNoWhitespace = this.stats.toJSONString();
			ScriptEngineManager manager = new ScriptEngineManager();
			ScriptEngine scriptEngine = manager.getEngineByName("JavaScript");
			scriptEngine.put("jsonString", jsonStringNoWhitespace);
			scriptEngine.eval("result = JSON.stringify(JSON.parse(jsonString), null, 2)");
			String prettyPrintedJson = (String) scriptEngine.get("result");
			
			outputWriter.write(prettyPrintedJson);			
			outputWriter.close();
		}
		System.out.println("SPICE evaluation took: " + timer.stop());
	}
}
