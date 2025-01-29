def reranking_rsa(summaries: pd.DataFrame, model, tokenizer, device):
    best_summaries = []
    
    for name, group in tqdm(summaries.groupby(["id"])):
        print(f"\nProcessing ID: {name} - Total candidates: {len(group)}")
        
        rsa_reranker = RSAReranking(
            model, tokenizer, device, 
            group.summary.unique().tolist(), 
            group.text.unique().tolist()
        )
        
        # Ottieni i punteggi RSA per tutte le frasi candidate
        best_rsa, best_base, speaker_df, listener_df, initial_listener, language_model_proba_df = rsa_reranker.rerank(t=3)
        
        # Converti speaker_df in DataFrame e associa i punteggi alle frasi candidate
        speaker_df = speaker_df.reset_index().rename(columns={'index': 'summary', 0: 'speaker_proba'})
        group = group.merge(speaker_df, on="summary", how="left")

        # Ordina le frasi per punteggio RSA (decrescente)
        group_sorted = group.sort_values(by="speaker_proba", ascending=False)

        # Seleziona il 50% migliore delle frasi
        num_selected = len(group_sorted) // 2
        top_50_percent = group_sorted.iloc[:num_selected]

        print(f"Selected {num_selected} best summaries for ID {name}")
        
        # Aggiungi ai risultati
        best_summaries.append(top_50_percent)

    # Unisci tutti i risultati in un unico DataFrame
    best_summaries = pd.concat(best_summaries)
    
    return best_summaries
