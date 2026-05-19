from django.db import models

# Create your models here.
"""
QueryRun model - stores every query submitted through the web interface.
linked to celer task via_id so we can poll status and retrieve results
"""

class QueryRun(models.Model):
    """
    Represent the store query submitted by the users.
    create immediate when query is submitted.
    updated when celery task completed. 
    """
    class Status(models.TextChoices) :
        PENDING = "PENDING", "Pending"
        STARTED = 'STARTED', "Running"
        SUCCESS = "SUCCESS", "Complete"
        FAILURE = "FAILURE", "Failed"
    # <====Core Fields=====>
    query_text = models.TextField(help_text = "Raw English legal query")
    expanded_query = models.TextField(blank = True, help_text = "German+English expanded query")
    task_id = models.CharField(max_length = 255, unique = True, db_index = True)
    status = models.CharField(max_length = 20, choices = Status.choices, default = Status.PENDING)

    # <====Result====>
    # store as a json string
    citations_json = models.TextField(blank=True, help_text = "JSON list of predicted citations")
    results_json = models.TextField(blank = True, help_text = "JSON list of full reranked results")

    # <====Evaluation====>
    f1_score = models.FloatField(null = True, blank=True)
    precision_score = models.FloatField(null = True, blank=True)
    recall_score = models.FloatField(null = True, blank = True)

    # <====pipeline setting used for this run====>
    hybrid_tor_k = models.IntegerField(default = 100)
    reranker_top_k = models.IntegerField(default = 10)
    use_stage2 = models.BooleanField(default = True)

    # <====Timing====>
    created_at = models.DateTimeField(auto_now_add=False)
    completed_at = models.DateTimeField(null = True, blank = True)
    elapsed_secs = models.FloatField(null= True, blank=True)

    # <====Error====>
    error_message = models.TextField(blank=True)



    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Query Run"

    def __str__(self):
        return f"[{self.status}] {self.query_text[:60]} ({self.created_at:%Y-%m-%d %H:%M})"

    @property
    def is_complete(self):
        return self.status == self.Status.SUCCESS
 
    @property
    def is_failed(self):
        return self.status == self.Status.FAILURE
 
    @property
    def duration_display(self):
        if self.elapsed_secs is None:
            return "—"
        mins = int(self.elapsed_secs // 60)
        secs = int(self.elapsed_secs % 60)
        return f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
 


