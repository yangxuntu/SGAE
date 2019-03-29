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
import java.util.Collections;
import java.util.Iterator;

import org.json.simple.JSONArray;
import org.json.simple.JSONAware;

public class TupleSet implements JSONAware {	
	
	protected ArrayList<SemanticTuple> tuples;
	public class Count {
		public int n;
		public double weighted_n;
		
		public Count(){
			n = 0;
			weighted_n = 0;
		}
	}
	
	public TupleSet(SceneGraph sg) {
		this(sg, TupleFilter.noFilter);
	}

	public TupleSet(SceneGraph sg, TupleFilter filter) {
		this.tuples = new ArrayList<SemanticTuple>();
		SemanticTuple proposedTuple;
		for (SceneGraphNode node : sg.nodeList()) {
			proposedTuple = new SemanticTuple(node.getObject());
			if (filter.operation(proposedTuple)) {
				this.tuples.add(proposedTuple);
			}
			for (SemanticConcept attr : node.getAttributes()) {
				proposedTuple = new SemanticTuple(node.getObject(), attr);
				if (filter.operation(proposedTuple)) {
					this.tuples.add(proposedTuple);
				}
			}
		}
		for (SceneGraphRelation edge : sg.relationList()) {
			proposedTuple = new SemanticTuple(edge.getSource().getObject(),
					edge.getRelation(), edge.getTarget().getObject());
			if (filter.operation(proposedTuple)) {
				this.tuples.add(proposedTuple);
			}
		}
		// Ensure repeatability
		Collections.sort(this.tuples);
	}
	
	public boolean add(SemanticTuple tuple){
		return this.tuples.add(tuple);
	}

	public int size() {
		return this.tuples.size();
	}

	public Count match_similar(TupleSet o) {
		Count count = new Count();
		for (SemanticTuple tup : o.tuples) {
			tup.truthValue = false;
		}
		for (SemanticTuple tup1 : this.tuples) {
			tup1.truthValue = false;
			for (SemanticTuple tup2 : o.tuples) {
				if (tup1.similarTo(tup2)) {
					tup1.truthValue = true;
					tup2.truthValue = true;
					count.n += 1;
					count.weighted_n += (tup1.idf + tup2.idf)/2.0;
					break;
				}
			}
		}
		// Ensure ordered output
		Collections.sort(this.tuples);		
		Collections.sort(o.tuples);
		return count;
	}
	
	public double weightedSize() {
		double result = 0.0;
		for (SemanticTuple t : tuples){
			result += t.idf;
		}
		return result;
	}
	
	public Count match_exact(TupleSet o) {
		Count count = new Count();
		for (SemanticTuple tup : o.tuples) {
			tup.truthValue = false;
		}
		for (SemanticTuple tup1 : this.tuples) {
			tup1.truthValue = false;
			for (SemanticTuple tup2 : o.tuples) {
				if (tup1.matchesTo(tup2)) {
					tup1.truthValue = true;
					tup2.truthValue = true;
					count.n += 1;
					count.weighted_n += (tup1.idf + tup2.idf)/2.0;
					break;
				}
			}
		}
		// Ensure ordered output
		Collections.sort(this.tuples);
		Collections.sort(o.tuples);
		return count;
	}

	@Override
	public String toJSONString() {
		return JSONArray.toJSONString(this.tuples);
	}

	@Override
	public String toString() {
		StringBuilder value_list = new StringBuilder();
		value_list.append("(");
		for (Iterator<SemanticTuple> iter = this.tuples.iterator(); iter.hasNext();) {
			value_list.append(iter.next());
			if (iter.hasNext()) {
				value_list.append(",");
			}
		}
		value_list.append(")");
		return value_list.toString();
	}

	public Object get(int i) {
		return this.tuples.get(i);
	}
}
