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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import edu.anu.spice.SemanticTuple;


@FunctionalInterface
public interface TupleFilter {

	public final static Set<String> smallNumbers = new HashSet<String>(
			Arrays.asList("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"));

	public final static Set<String> commonColors = new HashSet<String>(
			Arrays.asList("black", "brown", "blue", "red", "yellow", "pink", "gray", "grey", "green", "dark", "light",
					"white", "cream", "orange", "purple", "maroon", "beige", "violet"));

	public final static Set<String> sizeAdjectives = new HashSet<String>(Arrays.asList("baby", "illimitable",
			"scrawny", "beefy", "immeasurable", "short", "big", "immense", "sizable", "bony", "infinitesimal",
			"skeletal", "boundless", "lanky", "skimpy", "brawny", "large", "skinny", "broad", "lean", "slender",
			"bulky", "life-size", "slim", "chunky", "limitless", "small", "colossal", "little", "squat", "compact",
			"mammoth", "stocky", "corpulent", "massive", "stout", "cosmic", "meager", "strapping", "cubby", "measly",
			"sturdy", "curvy", "microscopic", "tall", "elfin", "mini", "teensy", "emaciated", "miniature", "teeny",
			"endless", "minuscule", "teeny-tiny", "enormous", "minute", "teeny-weeny", "epic", "narrow", "thick",
			"expansive", "obese", "thickset", "extensive", "outsized", "thin", "fat", "oversize", "tiny", "fleshy",
			"overweight", "titanic", "full-size", "paltry", "towering", "gargantuan", "petite", "trifling", "gaunt",
			"pint-size", "trim", "giant", "plump", "tubby", "gigantic", "pocket-size", "undersized", "grand", "portly",
			"underweight", "great", "pudgy", "unlimited", "heavy", "puny", "vast", "hefty", "rotund", "wee", "huge",
			"scanty", "whopping", "hulking", "scraggy", "wide"));

	boolean operation(SemanticTuple tuple);

	static TupleFilter noFilter = (SemanticTuple tuple) -> true;
	
	static TupleFilter objectFilter = (SemanticTuple tuple) -> tuple.size() == 1;
	
	static TupleFilter attributeFilter = (SemanticTuple tuple) -> tuple.size() == 2;
	
	static TupleFilter relationFilter = (SemanticTuple tuple) -> tuple.size() == 3;
	
	static TupleFilter cardinalityFilter = (SemanticTuple tuple) -> tuple.size() == 2
			&& smallNumbers.contains(tuple.get(1).toString());
	
	static TupleFilter colorFilter = (SemanticTuple tuple) -> tuple.size() == 2
			&& commonColors.contains(tuple.get(1).toString());
	
	static TupleFilter sizeFilter = (SemanticTuple tuple) -> tuple.size() == 2
			&& sizeAdjectives.contains(tuple.get(1).toString());
}
