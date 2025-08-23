import re
import aiohttp

#Renvoie l'erreur de type C identifiée dans le texte en utilisant un schéma prédéfini comprenant toutes les erreurs courantes.
def extract_c_error(text: str) -> str:

    c_error_patterns = [
        # Erreurs de compilation/linking
        r"undefined reference to `[^']+'",
        r"undefined reference to \w+",
        r"multiple definition of `[^']+'",
        r"ld returned \d+ exit status",
        r"undefined symbol: [^\n]+",
        r"ld: symbol\(s\) not found(?: for architecture)?[^\n]*",
        r"relocation truncated to fit[^\n]*",
        r"cannot find -l\w+",
        r"no such file or directory[^\n]*\.(h|c|cpp)",
        
        # Erreurs de syntaxe/types
        r"error: expected [^\n]+ before [^\n]+",
        r"error: invalid type argument of [^\n]+",
        r"error: '[^']+' undeclared \(first use in this function\)",
        r"error: incompatible types when [^\n]+",
        r"error: assignment of read-only variable '[^']+'",
        r"error: lvalue required as [^\n]+",
        r"error: too few arguments to function '[^']+'",
        r"error: too many arguments to function '[^']+'",
        r"error: return makes integer from pointer without a cast",
        r"error: conflicting types for '[^']+'",
        r"error: dereferencing pointer to incomplete type",
        r"error: expected declaration specifiers before [^\n]+",
        r"error: storage size of '[^']+' isn't known",
        
        # Erreurs de préprocesseur
        r"error: #[^\n]+not found[^\n]*",
        r"fatal error: [^:]+\.h: No such file or directory",
        r"error: macro [^\n]+ passed \d+ arguments, but takes just \d",
        
        # Erreurs d'exécution
        r"segmentation fault",
        r"core dumped",
        r"stack smashing detected",
        r"double free or corruption",
        r"malloc: memory corruption",
        r"free\(\): invalid pointer",
        r"munmap_chunk\(\): invalid pointer",
        r"stack overflow",
        r"floating point exception",
        r"bus error",
        
        # Avertissements importants
        r"warning: implicit declaration of function '[^']+'",
        r"warning: format '[^']+' expects argument of type '[^']+', but argument \d+ has type '[^']+'",
        r"warning: comparison between pointer and integer",
        r"warning: function returns address of local variable",
        r"warning: unused variable '[^']+'",
        
        # Fichiers et lignes
        r"\.(c|cpp|h):\d+:\d+: (?:error|warning|note): [^\n]+",
        r"In file included from [^\n]+",
        r"make(?:\[\d+\])?: \*\*\* [^\n]+",
        
        # Erreurs spécifiques C++
        r"error: '.*' was not declared in this scope",
        r"error: expected ';' after class definition",
        r"error: no matching function for call to '[^']+'",
        r"error: template argument [^\n]+",
        r"error: expected primary-expression before [^\n]+",
        r"error: expected ',' or ';' before [^\n]+"
    ]
    for pattern in c_error_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None

# Nettoyage pour la recherche
def clean_query(query: str) -> str:
    query = query.lower()
    return re.sub(r'[^a-z0-9\s\+\#\*]', '', query).strip()

def is_c_error(text: str) -> bool:
    return extract_c_error(text) is not None

# Recherche StackOverflow sinon recherche Google
async def search_stackoverflow(text: str) -> str:
    """Recherche StackOverflow uniquement pour l'erreur extraite."""
    error_only = extract_c_error(text)
    if not error_only:
        return None

    cleaned_query = clean_query(error_only)

    async with aiohttp.ClientSession() as session:
        for accepted_only in [True, False]:
            params = {
                "order": "desc",
                "sort": "votes",
                "q": cleaned_query,
                "tagged": "c;c++",
                "site": "stackoverflow",
                "filter": "!-*jbN-o8P3E5",
                "answers": 1
            }
            if accepted_only:
                params["accepted"] = True

            try:
                async with session.get(
                    "https://api.stackexchange.com/2.3/search/advanced",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()
                    if "items" in data and data["items"]:
                        best_match = max(
                            data["items"], 
                            key=lambda x: (x.get("score", 0), x.get("answer_count", 0))
                        )
                        return best_match["link"]
            except Exception as e:
                print(f"⚠️ API error: {e}")

    # Fallback Google
    return f"https://www.google.com/search?q=site:stackoverflow.com+c+{cleaned_query.replace(' ', '+')}"

