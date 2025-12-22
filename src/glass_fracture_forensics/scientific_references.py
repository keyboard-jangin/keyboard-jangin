#!/usr/bin/env python3
"""
SCIENTIFIC REFERENCES DATABASE
================================

Comprehensive bibliography and citation system for forensic analysis.
All methods and constants are traceable to peer-reviewed literature.

REFERENCE CATEGORIES:
1. Fracture Mechanics (LEFM)
2. Glass Material Properties
3. Computer Vision (Multi-view Geometry)
4. Statistical Methods
5. Forensic Science Standards

Author: Forensic Engineering Team
Version: 2.2
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ReferenceType(Enum):
    """Types of scientific references"""
    TEXTBOOK = "textbook"
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    STANDARD = "standard"
    TECHNICAL_REPORT = "technical_report"


@dataclass
class ScientificReference:
    """
    Single scientific reference with full citation
    """
    ref_id: str
    authors: List[str]
    title: str
    year: int
    ref_type: ReferenceType

    # Publication details
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None

    # Annotations
    topics: List[str] = field(default_factory=list)
    key_equations: List[str] = field(default_factory=list)
    notes: str = ""

    def cite_apa(self) -> str:
        """Generate APA citation"""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += ", et al."

        citation = f"{authors_str} ({self.year}). {self.title}."

        if self.journal:
            citation += f" {self.journal}"
            if self.volume:
                citation += f", {self.volume}"
            if self.pages:
                citation += f", {self.pages}"
        elif self.publisher:
            citation += f" {self.publisher}."

        if self.doi:
            citation += f" https://doi.org/{self.doi}"

        return citation

    def cite_ieee(self) -> str:
        """Generate IEEE citation"""
        authors_str = ", ".join([a.split()[-1] for a in self.authors[:3]])
        if len(self.authors) > 3:
            authors_str += ", et al."

        citation = f"{authors_str}, \"{self.title},\""

        if self.journal:
            citation += f" {self.journal}"
            if self.volume:
                citation += f", vol. {self.volume}"
            if self.pages:
                citation += f", pp. {self.pages}"

        citation += f", {self.year}."

        return citation


# ============================================================================
# FRACTURE MECHANICS REFERENCES
# ============================================================================

FRACTURE_MECHANICS_REFS = {
    "anderson2017": ScientificReference(
        ref_id="anderson2017",
        authors=["Anderson, T. L."],
        title="Fracture Mechanics: Fundamentals and Applications",
        year=2017,
        ref_type=ReferenceType.TEXTBOOK,
        publisher="CRC Press",
        isbn="978-1-4987-2813-3",
        topics=["LEFM", "stress_intensity_factors", "fracture_toughness"],
        key_equations=[
            "K_I = σ√(πa)f(geometry)",
            "K_Ic critical value for crack propagation"
        ],
        notes="Standard reference for linear elastic fracture mechanics. "
              "Contains derivations for stress intensity factors and J-integral."
    ),

    "lawn1993": ScientificReference(
        ref_id="lawn1993",
        authors=["Lawn, B. R."],
        title="Fracture of Brittle Solids",
        year=1993,
        ref_type=ReferenceType.TEXTBOOK,
        publisher="Cambridge University Press",
        isbn="978-0-521-40972-8",
        topics=["brittle_fracture", "crack_propagation", "glass"],
        key_equations=[
            "v = v0 exp(-Q/RT) for crack velocity",
            "γ = K²/E for surface energy"
        ],
        notes="Comprehensive treatment of brittle fracture in ceramics and glass. "
              "Essential for understanding crack branching and arrest."
    ),

    "freund1990": ScientificReference(
        ref_id="freund1990",
        authors=["Freund, L. B."],
        title="Dynamic Fracture Mechanics",
        year=1990,
        ref_type=ReferenceType.TEXTBOOK,
        publisher="Cambridge University Press",
        isbn="978-0-521-30330-5",
        topics=["dynamic_fracture", "crack_speed", "wave_propagation"],
        key_equations=[
            "v_crack < v_Rayleigh ≈ 0.9·v_shear",
            "Dynamic stress intensity K_dyn(v)"
        ],
        notes="Dynamic effects in fast crack propagation. "
              "Relevant for impact-induced fractures."
    ),

    "irwin1957": ScientificReference(
        ref_id="irwin1957",
        authors=["Irwin, G. R."],
        title="Analysis of Stresses and Strains Near the End of a Crack Traversing a Plate",
        year=1957,
        ref_type=ReferenceType.JOURNAL_ARTICLE,
        journal="Journal of Applied Mechanics",
        volume="24",
        pages="361-364",
        topics=["crack_tip_fields", "mode_I_II_III"],
        key_equations=[
            "σ_ij = K/(√2πr) f_ij(θ)",
            "Three fracture modes: I, II, III"
        ],
        notes="Foundational paper establishing stress intensity factor concept. "
              "Nobel Prize caliber work."
    ),
}


# ============================================================================
# GLASS MATERIAL PROPERTIES REFERENCES
# ============================================================================

GLASS_PROPERTIES_REFS = {
    "astm_c1036": ScientificReference(
        ref_id="astm_c1036",
        authors=["ASTM International"],
        title="ASTM C1036-16: Standard Specification for Flat Glass",
        year=2016,
        ref_type=ReferenceType.STANDARD,
        publisher="ASTM International",
        doi="10.1520/C1036-16",
        topics=["glass_properties", "standards", "soda_lime_glass"],
        key_equations=[],
        notes="Industry standard for flat glass specifications. "
              "Provides material property ranges for forensic reference."
    ),

    "wiederhorn1969": ScientificReference(
        ref_id="wiederhorn1969",
        authors=["Wiederhorn, S. M."],
        title="Fracture Surface Energy of Glass",
        year=1969,
        ref_type=ReferenceType.JOURNAL_ARTICLE,
        journal="Journal of the American Ceramic Society",
        volume="52",
        pages="99-105",
        doi="10.1111/j.1151-2916.1969.tb13350.x",
        topics=["surface_energy", "fracture_toughness"],
        key_equations=[
            "γ ≈ 3.5 J/m² for soda-lime glass"
        ],
        notes="Experimental determination of glass fracture surface energy. "
              "Critical for K_Ic calculations."
    ),

    "varshneya2006": ScientificReference(
        ref_id="varshneya2006",
        authors=["Varshneya, A. K."],
        title="Fundamentals of Inorganic Glasses",
        year=2006,
        ref_type=ReferenceType.TEXTBOOK,
        publisher="Academic Press",
        isbn="978-0-12-714970-7",
        topics=["glass_structure", "mechanical_properties"],
        notes="Comprehensive glass science textbook. "
              "Essential for understanding glass behavior under stress."
    ),
}


# ============================================================================
# COMPUTER VISION REFERENCES
# ============================================================================

COMPUTER_VISION_REFS = {
    "hartley2004": ScientificReference(
        ref_id="hartley2004",
        authors=["Hartley, R.", "Zisserman, A."],
        title="Multiple View Geometry in Computer Vision",
        year=2004,
        ref_type=ReferenceType.TEXTBOOK,
        publisher="Cambridge University Press",
        isbn="978-0-521-54051-3",
        topics=["multi_view_geometry", "essential_matrix", "triangulation"],
        key_equations=[
            "x₂ᵀ E x₁ = 0 (epipolar constraint)",
            "E = [t]_× R (essential matrix)",
            "X = triangulate(x₁, x₂, P₁, P₂)"
        ],
        notes="THE reference for multi-view geometry. "
              "All 3D reconstruction algorithms based on this work."
    ),

    "lucas1981": ScientificReference(
        ref_id="lucas1981",
        authors=["Lucas, B. D.", "Kanade, T."],
        title="An Iterative Image Registration Technique with an Application to Stereo Vision",
        year=1981,
        ref_type=ReferenceType.CONFERENCE_PAPER,
        journal="Proceedings of IJCAI",
        pages="674-679",
        topics=["optical_flow", "feature_tracking"],
        key_equations=[
            "Lucas-Kanade iterative optimization"
        ],
        notes="Original KLT tracking algorithm. "
              "Foundation for all feature tracking in this system."
    ),

    "kalal2010": ScientificReference(
        ref_id="kalal2010",
        authors=["Kalal, Z.", "Mikolajczyk, K.", "Matas, J."],
        title="Forward-Backward Error: Automatic Detection of Tracking Failures",
        year=2010,
        ref_type=ReferenceType.CONFERENCE_PAPER,
        journal="Proceedings of ICPR",
        pages="2756-2759",
        doi="10.1109/ICPR.2010.675",
        topics=["tracking_validation", "forward_backward_error"],
        notes="FB error for tracking validation - used in our system."
    ),
}


# ============================================================================
# STATISTICAL METHODS REFERENCES
# ============================================================================

STATISTICAL_REFS = {
    "efron1993": ScientificReference(
        ref_id="efron1993",
        authors=["Efron, B.", "Tibshirani, R. J."],
        title="An Introduction to the Bootstrap",
        year=1993,
        ref_type=ReferenceType.TEXTBOOK,
        publisher="Chapman & Hall/CRC",
        isbn="978-0-412-04231-7",
        topics=["bootstrap", "resampling", "confidence_intervals"],
        key_equations=[
            "Bootstrap: resample with replacement",
            "Percentile CI: [θ_{α/2}, θ_{1-α/2}]"
        ],
        notes="Definitive reference for bootstrap methods. "
              "Used for our uncertainty quantification."
    ),

    "rousseeuw1987": ScientificReference(
        ref_id="rousseeuw1987",
        authors=["Rousseeuw, P. J.", "Leroy, A. M."],
        title="Robust Regression and Outlier Detection",
        year=1987,
        ref_type=ReferenceType.TEXTBOOK,
        publisher="Wiley",
        isbn="978-0-471-85233-9",
        topics=["robust_statistics", "outlier_detection", "mahalanobis_distance"],
        notes="Standard reference for robust statistical methods."
    ),

    "james2013": ScientificReference(
        ref_id="james2013",
        authors=["James, G.", "Witten, D.", "Hastie, T.", "Tibshirani, R."],
        title="An Introduction to Statistical Learning",
        year=2013,
        ref_type=ReferenceType.TEXTBOOK,
        publisher="Springer",
        isbn="978-1-4614-7138-7",
        doi="10.1007/978-1-4614-7138-7",
        topics=["cross_validation", "model_selection"],
        notes="Modern statistical learning - CV methods used in validation."
    ),
}


# ============================================================================
# FORENSIC SCIENCE REFERENCES
# ============================================================================

FORENSIC_REFS = {
    "astm_e678": ScientificReference(
        ref_id="astm_e678",
        authors=["ASTM International"],
        title="ASTM E678-07: Standard Practice for Evaluation of Scientific or Technical Data",
        year=2007,
        ref_type=ReferenceType.STANDARD,
        publisher="ASTM International",
        doi="10.1520/E0678-07",
        topics=["forensic_standards", "evidence_evaluation"],
        notes="Standard practice for forensic data evaluation."
    ),

    "iso17025": ScientificReference(
        ref_id="iso17025",
        authors=["ISO"],
        title="ISO/IEC 17025:2017 - General Requirements for Competence of Testing and Calibration Laboratories",
        year=2017,
        ref_type=ReferenceType.STANDARD,
        publisher="International Organization for Standardization",
        topics=["laboratory_standards", "quality_assurance"],
        notes="International standard for testing lab competence. "
              "Ensures forensic results meet legal requirements."
    ),

    "thornton1997": ScientificReference(
        ref_id="thornton1997",
        authors=["Thornton, J. I."],
        title="The General Assumptions and Rationale of Forensic Identification",
        year=1997,
        ref_type=ReferenceType.BOOK_CHAPTER,
        journal="Modern Scientific Evidence",
        publisher="West Publishing",
        topics=["forensic_principles", "evidence_interpretation"],
        notes="Fundamental principles of forensic science. "
              "Establishes framework for evidence interpretation."
    ),

    "daubert1993": ScientificReference(
        ref_id="daubert1993",
        authors=["Supreme Court of the United States"],
        title="Daubert v. Merrell Dow Pharmaceuticals, Inc., 509 U.S. 579",
        year=1993,
        ref_type=ReferenceType.LEGAL_CASE,
        topics=["legal_admissibility", "scientific_evidence"],
        notes="Establishes criteria for scientific evidence admissibility. "
              "Peer review, testing, error rates, general acceptance."
    ),
}


# ============================================================================
# REFERENCE DATABASE
# ============================================================================

class ReferenceDatabase:
    """
    Complete reference database for forensic system
    """

    def __init__(self):
        self.references = {}
        self._load_all_references()

    def _load_all_references(self):
        """Load all reference categories"""
        for ref_dict in [
            FRACTURE_MECHANICS_REFS,
            GLASS_PROPERTIES_REFS,
            COMPUTER_VISION_REFS,
            STATISTICAL_REFS,
            FORENSIC_REFS
        ]:
            self.references.update(ref_dict)

    def get_reference(self, ref_id: str) -> Optional[ScientificReference]:
        """Get reference by ID"""
        return self.references.get(ref_id)

    def get_by_topic(self, topic: str) -> List[ScientificReference]:
        """Get all references for a topic"""
        return [
            ref for ref in self.references.values()
            if topic in ref.topics
        ]

    def generate_bibliography(self,
                            ref_ids: List[str],
                            style: str = "apa") -> str:
        """
        Generate formatted bibliography

        Args:
            ref_ids: List of reference IDs to include
            style: Citation style ('apa' or 'ieee')

        Returns:
            Formatted bibliography string
        """
        bib = "REFERENCES\n" + "="*70 + "\n\n"

        for i, ref_id in enumerate(sorted(ref_ids), 1):
            ref = self.get_reference(ref_id)
            if ref is None:
                continue

            if style == "apa":
                citation = ref.cite_apa()
            elif style == "ieee":
                citation = f"[{i}] {ref.cite_ieee()}"
            else:
                citation = ref.cite_apa()

            bib += f"{citation}\n\n"

        return bib

    def trace_method_to_source(self, method_name: str) -> List[str]:
        """
        Trace a method to its scientific sources

        Returns list of reference IDs
        """
        method_references = {
            "essential_matrix": ["hartley2004"],
            "klt_tracking": ["lucas1981", "kalal2010"],
            "stress_intensity_factors": ["anderson2017", "irwin1957"],
            "fracture_toughness": ["lawn1993", "wiederhorn1969"],
            "bootstrap_ci": ["efron1993"],
            "outlier_detection": ["rousseeuw1987"],
            "cross_validation": ["james2013"],
            "glass_properties": ["astm_c1036", "varshneya2006"],
            "forensic_standards": ["astm_e678", "iso17025"],
            "evidence_admissibility": ["daubert1993"],
        }

        return method_references.get(method_name, [])


# ============================================================================
# TRACEABILITY DOCUMENTATION
# ============================================================================

def generate_traceability_matrix() -> Dict[str, List[str]]:
    """
    Generate traceability matrix: Code → Theory → References

    Returns:
        Dictionary mapping code components to references
    """
    return {
        # Core algorithms
        "RelativeReconstructor.reconstruct": [
            "hartley2004"  # Essential matrix, triangulation
        ],
        "FeatureTracker.track_features": [
            "lucas1981",   # KLT algorithm
            "kalal2010"    # Forward-backward error
        ],
        "OriginEstimator.estimate_origin": [
            "hartley2004"  # Multi-view geometry
        ],

        # Fracture mechanics
        "FractureMechanicsAnalyzer.compute_stress_intensity": [
            "anderson2017",  # LEFM theory
            "irwin1957"      # Stress intensity factors
        ],
        "GlassMaterialProperties": [
            "astm_c1036",    # Standard properties
            "varshneya2006", # Glass science
            "wiederhorn1969" # Fracture energy
        ],

        # Statistical methods
        "bootstrap_origin_estimation": [
            "efron1993"  # Bootstrap theory
        ],
        "robust_outlier_detection": [
            "rousseeuw1987"  # Robust statistics
        ],
        "leave_one_out_cross_validation": [
            "james2013"  # Cross-validation
        ],

        # Forensic compliance
        "ForensicReport.generate_hash": [
            "astm_e678",  # Evidence standards
            "iso17025"    # Laboratory standards
        ],
        "validate_system": [
            "daubert1993"  # Legal admissibility
        ],
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Demo reference system"""

    print("="*70)
    print("SCIENTIFIC REFERENCES DATABASE - DEMO")
    print("="*70)

    db = ReferenceDatabase()

    # 1. Get specific reference
    print("\n1. Example Reference (Anderson 2017):")
    print("-" * 70)

    ref = db.get_reference("anderson2017")
    if ref:
        print(f"APA: {ref.cite_apa()}\n")
        print(f"IEEE: {ref.cite_ieee()}\n")
        print(f"Topics: {', '.join(ref.topics)}")
        print(f"Key equations: {ref.key_equations}")

    # 2. References by topic
    print("\n2. LEFM References:")
    print("-" * 70)

    lefm_refs = db.get_by_topic("LEFM")
    for ref in lefm_refs:
        print(f"  - {ref.cite_apa()}")

    # 3. Method traceability
    print("\n3. Method Traceability:")
    print("-" * 70)

    method = "stress_intensity_factors"
    sources = db.trace_method_to_source(method)
    print(f"Method: {method}")
    print(f"Source references:")
    for ref_id in sources:
        ref = db.get_reference(ref_id)
        if ref:
            print(f"  - {ref.cite_apa()}")

    # 4. Generate bibliography
    print("\n4. Sample Bibliography:")
    print("-" * 70)

    selected_refs = ["anderson2017", "hartley2004", "efron1993"]
    bib = db.generate_bibliography(selected_refs, style="apa")
    print(bib)

    # 5. Traceability matrix
    print("\n5. Code-to-Theory Traceability:")
    print("-" * 70)

    matrix = generate_traceability_matrix()
    for code_component, ref_ids in list(matrix.items())[:3]:
        print(f"\n{code_component}:")
        for ref_id in ref_ids:
            ref = db.get_reference(ref_id)
            if ref:
                print(f"  → {ref.authors[0]} ({ref.year}): {ref.title[:50]}...")

    print("\n" + "="*70)
    print(f"Total references in database: {len(db.references)}")
    print("="*70)
