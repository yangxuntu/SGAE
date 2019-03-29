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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;


/**
 * Calculates SemanticTuple (inverse) document frequency based on the reference set.
 *
 */
public class DocumentFrequency {

	protected HashMap<String, Integer> df;
	protected int N;

	public DocumentFrequency() {
		this.df = new HashMap<String, Integer>();
		this.N = 0;
	}
	
	public void addTuples(Map<String, ArrayList<ArrayList<String>>> tuples){
		for (Entry<String, ArrayList<ArrayList<String>>> entry : tuples.entrySet()) {
			for (ArrayList<String> t : entry.getValue()){
				String tString = t.toString();
				if (this.df.containsKey(tString)){
					this.df.put(tString, this.df.get(tString)+1);
				} else {
					this.df.put(tString, 1);
				}
				this.N += 1;
			}
		}
	}

	public int df(SemanticTuple tuple) {
		ArrayList<ArrayList<String>> enumeratedTuples = tuple.enumerateTuples();
		int df = 0;
		// Examine all the enumerated tuples of a SemanticConcept
		for (ArrayList<String> t: enumeratedTuples){
			String tString = t.toString();
			if (this.df.containsKey(tString)){
				df += this.df.get(tString);
			}
		}
		return df;
	}

	public double idf(SemanticTuple tuple) {
		return Math.log( (double)N / Math.max(this.df(tuple),1) );
	}

}
