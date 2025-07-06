from django.db import models

# Create your models here.

class University(models.Model):
    rank_2026 = models.IntegerField(null=True, blank=True)
    rank_2025 = models.IntegerField(null=True, blank=True)
    institution_name = models.CharField(max_length=200)
    country = models.CharField(max_length=100)
    region = models.CharField(max_length=100)
    size = models.CharField(max_length=10)
    focus = models.CharField(max_length=10)
    research_intensity = models.CharField(max_length=10)
    status = models.CharField(max_length=50)
    
    # Score columns
    score_ar = models.FloatField(null=True, blank=True)  # Academic Reputation
    rank_ar = models.IntegerField(null=True, blank=True)
    score_er = models.FloatField(null=True, blank=True)  # Employer Reputation
    rank_er = models.IntegerField(null=True, blank=True)
    score_fs = models.FloatField(null=True, blank=True)  # Faculty Student Ratio
    rank_fs = models.IntegerField(null=True, blank=True)
    score_cpf = models.FloatField(null=True, blank=True)  # Citations per Faculty
    rank_cpf = models.IntegerField(null=True, blank=True)
    score_if = models.FloatField(null=True, blank=True)  # International Faculty
    rank_if = models.IntegerField(null=True, blank=True)
    score_is = models.FloatField(null=True, blank=True)  # International Students
    rank_is = models.IntegerField(null=True, blank=True)
    score_isd = models.FloatField(null=True, blank=True)  # International Student Diversity
    rank_isd = models.IntegerField(null=True, blank=True)
    score_irn = models.FloatField(null=True, blank=True)  # International Research Network
    rank_irn = models.IntegerField(null=True, blank=True)
    score_eo = models.FloatField(null=True, blank=True)  # Employment Outcomes
    rank_eo = models.IntegerField(null=True, blank=True)
    score_st = models.FloatField(null=True, blank=True)  # Sustainability
    rank_st = models.IntegerField(null=True, blank=True)
    overall_score = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return self.institution_name
    
    class Meta:
        ordering = ['rank_2026']
