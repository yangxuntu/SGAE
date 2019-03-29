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

/**
*
* Based on the SceneGraphRelation in scenegraph-1.0 by Sebastian Schuster.
*
*/

public class SceneGraphRelation implements java.io.Serializable {

	private static final long serialVersionUID = 1L;
	private final SceneGraphNode source;
	private final SceneGraphNode target;
	private SemanticConcept relation;

	public SceneGraphRelation(SceneGraphNode source, SceneGraphNode target, SemanticConcept relation) {
		this.source = source;
		this.target = target;
		this.relation = relation;
	}

	public SceneGraphNode getSource() {
		return source;
	}

	public SceneGraphNode getTarget() {
		return target;
	}

	public SemanticConcept getRelation() {
		return relation;
	}

	@Override
	public int hashCode() {
		return new int[] { this.source.hashCode(), this.target.hashCode(), this.relation.hashCode() }.hashCode();
	}

	@Override
	public boolean equals(Object o) {
		if (o == null) {
			return false;
		}
		if (!(o instanceof SceneGraphRelation)) {
			return false;
		}
		SceneGraphRelation oReln = (SceneGraphRelation) o;
		return this.source.equals(oReln.source) && this.target.equals(oReln.target)
				&& this.relation.equals(oReln.relation);
	}

	public boolean similarTo(Object o) {
		if (o == null) {
			return false;
		}
		if (!(o instanceof SceneGraphRelation)) {
			return false;
		}
		SceneGraphRelation oReln = (SceneGraphRelation) o;
		// Must have same objects and a similar relation
		return this.source.equals(oReln.source) && this.target.equals(oReln.target)
				&& this.relation.similarTo(oReln.relation);
	}

	public boolean merge(SceneGraphRelation oReln) {
		if (!this.source.equals(oReln.source)) {
			return false;
		}
		if (!this.target.equals(oReln.target)) {
			return false;
		}
		return this.relation.merge(oReln.relation);
	}

}
