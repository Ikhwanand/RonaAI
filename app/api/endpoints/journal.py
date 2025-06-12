from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.journals import Journals
from app.models.users import User 
from app.schemas.journal import JournalCreate
from app.schemas.responses import APIResponse
from app.core.security import get_current_user

router = APIRouter()


@router.post("/create-journal", response_model=APIResponse)
async def create_journal(journal_data: JournalCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new journal entry."""
    journal = Journals(
        user_id=current_user.id,
        title=journal_data.title,
        content=journal_data.content
    )
    db.add(journal)
    db.commit()
    db.refresh(journal)

    # Convert SQLAlchemy model to dictionary
    journal_dict = {
        "id": journal.id,
        "user_id": journal.user_id,
        "title": journal.title,
        "content": journal.content,
        "created_at": journal.created_at,
        "updated_at": journal.updated_at
    }

    return APIResponse(
        success=True,
        message="Journal created successfully",
        data=journal_dict
    )


@router.get("/get-journals", response_model=APIResponse)
async def get_journals(current_user: User = Depends(get_current_user), db: Session = Depends(get_db), skip: int = 0, limit: int = 10):
    """Get user's journal entries with pagination and optional mood filter"""
    query = db.query(Journals).filter(Journals.user_id == current_user.id)

    total = query.count()
    journals = query.order_by(Journals.created_at.desc()).offset(skip).limit(limit).all()

    # Convert SQLAlchemy models to dictionaries
    journals_list = []
    for journal in journals:
        journals_list.append({
            "id": journal.id,
            "user_id": journal.user_id,
            "title": journal.title,
            "content": journal.content,
            "created_at": journal.created_at,
            "updated_at": journal.updated_at
        })

    return APIResponse(
        success=True,
        message="Journals retrieved successfully",
        data={
            "items": journals_list,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    )


@router.get("/get-journal/{journal_id}", response_model=APIResponse)
async def get_journal(journal_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get a specific journal entry"""
    journal = db.query(Journals).filter(
        Journals.id == journal_id,
        Journals.user_id == current_user.id
    ).first()

    if not journal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal not found"
        )
    
    # Convert SQLAlchemy model to dictionary
    journal_dict = {
        "id": journal.id,
        "user_id": journal.user_id,
        "title": journal.title,
        "content": journal.content,
        "created_at": journal.created_at,
        "updated_at": journal.updated_at
    }

    return APIResponse(
        success=True,
        message="Journal retrieved successfully",
        data=journal_dict
    )


@router.put("/update-journal/{journal_id}", response_model=APIResponse)
async def update_journal(
    journal_id: int,
    journal_data: JournalCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a journal entry"""
    journal = db.query(Journals).filter(
        Journals.id == journal_id,
        Journals.user_id == current_user.id
    ).first()

    if not journal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Journal not found")

    
    journal.title = journal_data.title 
    journal.content = journal_data.content 

    db.commit()
    db.refresh(journal)

    # Convert SQLAlchemy model to dictionary
    journal_dict = {
        "id": journal.id,
        "user_id": journal.user_id,
        "title": journal.title,
        "content": journal.content,
        "created_at": journal.created_at,
        "updated_at": journal.updated_at
    }

    return APIResponse(
        success=True,
        message="Journal updated successfully",
        data=journal_dict
    )


@router.delete("/delete-journal/{journal_id}", response_model=APIResponse)
async def delete_journal(journal_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """ Delete a journal entry"""
    journal = db.query(Journals).filter(
        Journals.id == journal_id,
        Journals.user_id == current_user.id
    ).first()

    if not journal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal not found"
        )
    
    db.delete(journal)
    db.commit()

    return APIResponse(
        success=True,
        message="Journal deleted successfully"
    )

    