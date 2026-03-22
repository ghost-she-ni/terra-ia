# PROJECT_SCOPE — Terra-IA

## Version officielle du projet

**Terra-IA est un pilote IA de scoring de constructibilité morphologique sur la commune de Chambéry (73065), à partir des données LiDAR HD IGN, comparant une baseline déterministe (CPI) à un modèle machine learning explicable (XGBoost + SHAP) pour le pré-filtrage foncier.**

C'est **la version exacte du projet soutenue au Capstone**.

---

## 1. Problème adressé

Dans le contexte de la densification urbaine et du ZAN, identifier des parcelles à potentiel est encore largement réalisé de manière manuelle, lente, coûteuse et peu reproductible. Or, la constructibilité d'un terrain ne dépend pas uniquement des règles d'urbanisme : elle dépend aussi de contraintes **physiques et morphologiques** observables sur le terrain.

Terra-IA répond à ce besoin en évaluant automatiquement, à l'échelle parcellaire, des facteurs comme la pente, l'humidité potentielle, la rugosité, l'ensoleillement ou l'enclavement à partir du **LiDAR HD IGN à 50 cm**.

**Important : Terra-IA ne prédit pas la constructibilité réglementaire.** Le projet mesure une **constructibilité morphologique**, complémentaire aux règles du PLU et aux servitudes.

---

## 2. Valeur business

Terra-IA vise à accélérer le **pré-filtrage foncier** et la **priorisation des parcelles** à étudier.

### Utilisateurs cibles
- Promoteurs immobiliers : prospection foncière intra-urbaine
- Collectivités locales : planification de la densification
- Foncières / investisseurs : tri initial d'opportunités
- Bureaux d'études : pré-qualification rapide de sites

### Valeur apportée
- Réduction du temps d'analyse initiale
- Approche plus systématique et reproductible
- Lecture physique du terrain absente des outils réglementaires classiques
- Explication des scores grâce à SHAP

### Positionnement
Le projet n'a **pas vocation à prendre la décision finale**. Il sert à répondre à la question :

> **Cette parcelle mérite-t-elle qu'on ouvre un dossier d'étude plus approfondi ?**

---

## 3. Périmètre officiel soutenu

### Zone étudiée
- **Commune pilote : Chambéry (INSEE 73065)**
- Emprise opérationnelle actuelle : **centre de Chambéry sur une emprise LiDAR partielle d'environ 2 km × 2 km**

### Données incluses
- IGN LiDAR HD (MNT / MNH, résolution 50 cm)
- Parcelles cadastrales
- DVF 2023 pour enrichissement / weak supervision

### Sorties incluses dans la version officielle
- Extraction de features morphologiques par parcelle
- Filtrage des parcelles hors périmètre ou non pertinentes
- Baseline déterministe : **CPI (Constructibility Potential Index)**
- Modèle ML de ranking/tabulaire : **XGBoost LambdaMART**
- Explicabilité globale : **SHAP**
- Validation spatiale documentée

### Chiffres officiels à utiliser
- **15 480 parcelles cadastrales**
- **2 536 parcelles valides**
- **2 315 parcelles labellisées**
- **10 features LiDAR ML**

Ces chiffres sont ceux des artefacts V3 actuellement disponibles et servent de référence pour le rapport, le README, les slides et la démo.

---

## 4. Ce qui est explicitement hors périmètre

Les éléments suivants **ne font pas partie du périmètre officiel soutenu**, sauf s'ils sont réellement régénérés, validés et documentés avant le rendu final :

- Couverture complète de la commune avec **24 dalles** ou **70–80 % de parcelles valides**
- Intégration stabilisée de la distance à la route via **OSMnx**
- Intégration du **PLU**, des servitudes et de la constructibilité réglementaire
- Détection automatique de dents creuses à l'échelle nationale
- Traitement direct des nuages de points LiDAR bruts
- Deep learning 3D, segmentation sémantique ou modèles vision complexes
- Passage multi-communes / industrialisation nationale

Si ces points sont mentionnés, ils doivent apparaître comme **perspectives** ou **travaux futurs**, pas comme résultats acquis.

---

## 5. Limites connues

### Limites de couverture
- L'emprise raster actuelle est partielle ; une large part des parcelles cadastrales reste hors emprise exploitable.

### Limites de labels
- Les labels DVF restent partiels et la jointure DVF/cadastre doit être consolidée.

### Limites métier
- Le score produit un **signal de pré-filtrage**, pas une validation de faisabilité juridique, financière ou opérationnelle.

### Limites méthodologiques
- Une partie du problème semble fortement structurée par les variables de pente.
- Les bonnes performances ne doivent pas être surinterprétées comme une résolution complète du problème réel.

### Limites de données
- Le projet dépend de la disponibilité et de l'emprise des rasters IGN.
- Certaines briques prévues initialement (ex. accessibilité voirie) ne sont pas encore stabilisées dans les livrables actuels.

---

## 6. Décision de storytelling

À partir de maintenant, tous les supports doivent raconter la même histoire :

1. Terra-IA traite un vrai problème métier de **pré-filtrage foncier**.
2. Le projet apporte une lecture **morphologique** du terrain.
3. La baseline CPI sert de référence.
4. Le ML est utilisé pour démontrer une **valeur ajoutée mesurable et explicable**.
5. Le projet soutenu est un **pilote robuste et honnête**, pas une plateforme nationale déjà finalisée.

---

## 7. Phrase officielle à reprendre partout

> **Terra-IA is an AI-assisted morphological buildability scoring pilot for Chambéry, using IGN LiDAR HD data to compare a deterministic baseline with an explainable machine learning model for land pre-screening.**

Version française :

> **Terra-IA est un pilote IA de scoring de constructibilité morphologique sur Chambéry, fondé sur le LiDAR HD IGN, qui compare une baseline déterministe à un modèle ML explicable pour le pré-filtrage foncier.**
